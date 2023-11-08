# -*- coding: utf-8 -*-
"""
Created on Wed April 03 14:27:26 2019

Description: Controller class implementations for:

https://www.frontiersin.org/articles/10.3389/fnins.2020.00166/

@author: John Fleming, john.fleming@ucdconnect.ie
"""

import math
import numpy as np
import scipy.signal as signal
from mpi4py import MPI
from numpy.linalg import LinAlgError

rank = MPI.COMM_WORLD.Get_rank()


class ZeroController:
    """Dummy controller with no stimulation"""

    def __init__(self, setpoint=0.0, ts=0.02):

        self.setpoint = setpoint
        self.label = "ZeroController"

        self.ts = ts
        self.current_time = 0.0  # (sec)
        self.last_time = 0.0

        # Initialize controller terms
        self.last_error = 0.0
        self.last_output_value = 0.0

        # Initialize the output value of the controller
        self.output_value = 0.0

        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

    def clear(self):
        """Clears controller variables"""

        self.last_error = 0.0

        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

        self.output_value = 0.0

    def update(self, state_value, current_time):
        """Update controller state"""

        # Calculate Error - if SetPoint > 0.0, then normalize error with
        # respect to set point
        if self.setpoint == 0.0:
            error = state_value - self.setpoint
        else:
            error = (state_value - self.setpoint) / self.setpoint

        # Converting from msec to sec
        self.current_time = current_time / 1000.0

        # Remember last time and last error for next calculation
        self.last_time = self.current_time
        self.last_error = error

        self.output_value = 0

        # Update the last output value
        self.last_output_value = self.output_value

        # Record state, error, y(t), and sample time values
        self.state_history.append(state_value)
        self.error_history.append(error)
        self.output_history.append(self.output_value)
        # Convert from msec to sec
        self.sample_times.append(current_time / 1000)

        # Return controller output
        return self.output_value

    def generate_dbs_signal(
        self,
        start_time,
        stop_time,
        dt,
        amplitude,
        frequency,
        pulse_width,
        offset,
        last_pulse_time_prior=0,
    ):
        return generate_monophasic_square_dbs_signal(
            start_time,
            stop_time,
            dt,
            amplitude=0,
            frequency=0,
            pulse_width=pulse_width,
            offset=offset,
            last_pulse_time_prior=last_pulse_time_prior
            )


class ConstantController:
    """Constant DBS Parameter Controller Class"""

    def __init__(
        self,
        setpoint=0.0,
        min_value=0.0,
        max_value=1e9,
        stimulation_amplitude=0.0,
        ts=0.0,
        units="mA",
    ):
        # Initial Controller Values
        self.setpoint = setpoint
        self.max_value = max_value
        self.min_value = min_value
        self.stimulation_amplitude = stimulation_amplitude
        self.ts = ts  # should be in sec as per above
        self.units = units
        self.label = "Constant_Controller/%f%s" % (self.stimulation_amplitude, self.units)

        # Set output value
        self.output_value = 0

        # Lists for tracking controller history
        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

    def clear(self):
        """Clears current On-Off controller output value and history"""

        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

        self.output_value = 0.0

    def update(self, state_value, current_time):
        """Calculates biomarker for constant DBS value

        u = self.ConstantValue

        """

        # Calculate Error - if SetPoint > 0.0
        # normalize error with respect to set point
        if self.setpoint == 0.0:
            error = state_value - self.setpoint
        else:
            error = (state_value - self.setpoint) / self.setpoint

        # Bound the controller output (between MinValue - MaxValue)
        if self.stimulation_amplitude > self.max_value:
            self.output_value = self.max_value
        elif self.stimulation_amplitude < self.min_value:
            self.output_value = self.min_value
        else:
            self.output_value = self.stimulation_amplitude

        # Record state, error and sample time values
        self.state_history.append(state_value)
        self.error_history.append(error)
        self.output_history.append(self.output_value)
        # Convert from msec to sec
        self.sample_times.append(current_time / 1000)

        return self.output_value

    def generate_dbs_signal(
        self,
        start_time,
        stop_time,
        dt,
        amplitude,
        frequency,
        pulse_width,
        offset,
        last_pulse_time_prior=0,
    ):
        return generate_monophasic_square_dbs_signal(
            start_time,
            stop_time,
            dt,
            amplitude,
            frequency,
            pulse_width,
            offset,
            last_pulse_time_prior
            )


class OnOffController:
    """On-Off Controller Class"""

    def __init__(
        self, setpoint=0.0, min_value=0.0, max_value=1e9, ramp_duration=0.25, ts=0.02
    ):
        # Initial Controller Values
        self.setpoint = setpoint
        self.max_value = max_value
        self.min_value = min_value
        # should be defined in sec, i.e. 0.25 sec
        self.ramp_duration = ramp_duration
        # should be in sec as per above
        self.ts = ts
        self.label = "On_Off_Controller"

        # Calculate how much controller output value will change each
        # controller call
        self.output_value_increment = (self.max_value - self.min_value) / math.ceil(
            self.ramp_duration / self.ts
        )

        # Initialize the output value of the controller
        self.last_output_value = 0
        self.output_value = 0

        # Lists for tracking controller history
        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

    def clear(self):
        """Clears current On-Off controller output value and history"""

        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

        self.last_output_value = 0.0
        self.output_value = 0.0

    def update(self, state_value, current_time):
        """Calculates updated controller output value for given reference feedback

        y(t) = y(t-1) + u(t)

        where:

        u(t) = MaxValue / (RampDuration/Ts) if e(t) > SetPoint
        or -MaxValue / (RampDuration/Ts) if e(t) < SetPoint

        """

        # Calculate Error - if SetPoint > 0.0
        # normalize error withrespect to set point
        if self.setpoint == 0.0:
            error = state_value - self.setpoint
            increment = 0.0
        else:
            error = (state_value - self.setpoint) / self.setpoint
            if error > 0.0:
                increment = self.output_value_increment
            else:
                increment = -self.output_value_increment

        # Bound the controller output (between MinValue - MaxValue)
        if self.last_output_value + increment > self.max_value:
            self.output_value = self.max_value
        elif self.last_output_value + increment < self.min_value:
            self.output_value = self.min_value
        else:
            self.output_value = self.last_output_value + increment

        # Record state, error and sample time values
        self.state_history.append(state_value)
        self.error_history.append(error)
        self.output_history.append(self.output_value)
        # Convert from msec to sec
        self.sample_times.append(current_time / 1000)

        self.last_output_value = self.output_value

        return self.output_value

    def set_max_value(self, max_value):
        """Sets the upper bound for the controller output"""
        self.max_value = max_value
        self.output_value_increment = (self.max_value - self.min_value) / (
            self.ramp_duration / self.ts
        )

    def set_min_value(self, min_value):
        """Sets the lower bound for the controller output"""
        self.min_value = min_value
        self.output_value_increment = (self.max_value - self.min_value) / (
            self.ramp_duration / self.ts
        )

    def set_ramp_duration(self, ramp_duration):
        """Sets how long the controller output takes to reach its max value"""
        self.ramp_duration = ramp_duration
        self.output_value_increment = (self.max_value - self.min_value) / (
            self.ramp_duration / self.ts
        )

    def set_ts(self, Ts):
        """Sets the sampling rate of the controller"""
        self.ts = Ts
        self.output_value_increment = (self.max_value - self.min_value) / (
            self.ramp_duration / self.ts
        )

    def set_label(self, label):
        """Sets the label of the controller"""
        self.label = label

    def set_setpoint(self, set_point):
        self.setpoint = set_point

    def get_state_history(self):
        return self.state_history

    def get_error_history(self):
        return self.error_history

    def get_output_history(self):
        return self.output_history

    def get_sample_times(self):
        return self.sample_times

    def get_label(self):
        return self.label


class DualThresholdController:
    """Dual-Threshold Controller Class"""

    def __init__(
        self,
        lower_threshold=0.0,
        upper_threshold=0.1,
        min_value=0.0,
        max_value=1e9,
        ramp_duration=0.25,
        ts=0.02,
    ):
        # Initial Controller Values
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.max_value = max_value
        self.min_value = min_value
        # should be defined in sec, i.e. 0.25 sec
        self.ramp_duration = ramp_duration
        # should be in sec as per above
        self.ts = ts
        self.label = "Dual_Threshold_Controller"

        # Calculate how much controller output value will change
        # each controller call
        self.output_value_increment = (self.max_value - self.min_value) / math.ceil(
            self.ramp_duration / self.ts
        )

        # Initialize the output value of the controller
        self.last_output_value = 0.0
        self.output_value = 0.0

        # Lists for tracking controller history
        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

    def clear(self):
        """Clears current dual-threshold controller output value and history"""

        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

        self.last_output_value = 0.0
        self.output_value = 0.0

    def update(self, state_value, current_time):
        """Calculates updated controller output value for given reference feedback

        if state_value > upper_threshold:
            y(t) = y(t-1) + u(t)
        elif state_value < lower_threshold:
            y(t) = y(t-1) + u(t)
        else:
            y(t) = y(t-1)
        where:

        u(t) = MaxValue / (RampDuration / Ts)
        if state_value(t) > UpperThreshold
        or
        u(t) = -MaxValue / (RampDuration / Ts)
        if state_value(t) < LowerThreshold

        """

        # Check how to update controller value and calculate error with
        # respect to upper/lower threshold
        # Increase if above upper threshold
        if state_value > self.upper_threshold:
            error = (state_value - self.upper_threshold) / self.upper_threshold
            increment = self.output_value_increment
        # Decrease if below lower threshold
        elif state_value < self.lower_threshold:
            error = (state_value - self.lower_threshold) / self.lower_threshold
            increment = -self.output_value_increment
        # Do nothing when within upper and lower thresholds
        else:
            error = 0
            increment = 0

        # Bound the controller output (between MinValue - MaxValue)
        if self.last_output_value + increment > self.max_value:
            self.output_value = self.max_value
        elif self.last_output_value + increment < self.min_value:
            self.output_value = self.min_value
        else:
            self.output_value = self.last_output_value + increment

        # Record state, error and sample time values
        self.state_history.append(state_value)
        self.error_history.append(error)
        self.output_history.append(self.output_value)
        # Convert from msec to sec
        self.sample_times.append(current_time / 1000)

        self.last_output_value = self.output_value

        return self.output_value

    def set_upper_threshold(self, upper_threshold):
        """Sets the upper threshold for the measured state"""
        self.upper_threshold = upper_threshold

    def set_lower_threshold(self, lower_threshold):
        """Sets the lower threshold for the measured state"""
        self.lower_threshold = lower_threshold

    def set_max_value(self, max_value):
        """Sets the upper bound for the controller output"""
        self.max_value = max_value
        self.output_value_increment = (self.max_value - self.min_value) / (
            self.ramp_duration / self.ts
        )

    def set_min_value(self, min_value):
        """Sets the lower bound for the controller output"""
        self.min_value = min_value
        self.output_value_increment = (self.max_value - self.min_value) / (
            self.ramp_duration / self.ts
        )

    def set_ramp_duration(self, ramp_duration):
        """Sets how long the controller output takes to reach its max value"""
        self.ramp_duration = ramp_duration
        self.output_value_increment = (self.max_value - self.min_value) / (
            self.ramp_duration / self.ts
        )

    def set_ts(self, Ts):
        """Sets the sampling rate of the controller"""
        self.ts = Ts
        self.output_value_increment = (self.max_value - self.min_value) / (
            self.ramp_duration / self.ts
        )

    def set_label(self, label):
        """Sets the label of the controller"""
        self.label = label

    def get_state_history(self):
        return self.state_history

    def get_error_history(self):
        return self.error_history

    def get_output_history(self):
        return self.output_history

    def get_sample_times(self):
        return self.sample_times

    def get_label(self):
        return self.label


class StandardPIDController:
    """Standard PID Controller Class"""

    def __init__(
        self, setpoint=0.0, kp=0.0, ti=0.0, td=0.0, ts=0.02, min_value=0.0, max_value=1e9
    ):

        self.setpoint = setpoint
        self.kp = kp
        self.ti = ti
        self.td = td

        # Set output value bounds
        self.min_value = min_value
        self.max_value = max_value

        self.label = "standard_PID_Controller/Kp=%f, Ti=%f, Td=%f" % (
            self.kp,
            self.ti,
            self.td,
        )

        self.ts = ts
        self.current_time = 0.0  # (sec)
        self.last_time = 0.0

        # Initialize controller terms
        self.integral_term = 0.0
        self.differential_term = 0.0
        self.last_error = 0.0
        self.last_output_value = 0.0

        # Initialize the output value of the controller
        self.output_value = 0.0

        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

    def clear(self):
        """Clears PID computations and coefficients"""

        self.integral_term = 0.0
        self.differential_term = 0.0
        self.last_error = 0.0

        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

        self.output_value = 0.0

    def update(self, state_value, current_time):
        """Calculates controller output signal for given reference feedback

        where:

        u(t) = K_p (e(t) + (1/T_i)* \\int_{0}^{t} e(t)dt + T_d {de}/{dt})

        where the error calculated is the tracking error (r(t) - y(t))

        """

        # Calculate Error - if SetPoint > 0.0, then normalize error with
        # respect to set point
        if self.setpoint == 0.0:
            error = state_value - self.setpoint
        else:
            error = (state_value - self.setpoint) / self.setpoint

        # Converting from msec to sec
        self.current_time = current_time / 1000.0
        delta_time = self.ts
        delta_error = error - self.last_error

        self.integral_term += error * delta_time

        self.differential_term = 0.0
        if delta_time > 0:
            self.differential_term = delta_error / delta_time

        # Remember last time and last error for next calculation
        self.last_time = self.current_time
        self.last_error = error

        # Calculate u(t) - catch potential division by zero error
        try:
            u = self.kp * (
                error + ((1.0 / self.ti) * self.integral_term) + (self.td * self.differential_term)
            )
        except ZeroDivisionError:
            u = self.kp * (error + (0.0 * self.integral_term) + (self.td * self.differential_term))

        # Bound the controller output if necessary
        # (between MinValue - MaxValue)
        if u > self.max_value:
            self.output_value = self.max_value
            # Back-calculate the integral error
            self.integral_term -= error * delta_time
        elif u < self.min_value:
            self.output_value = self.min_value
            # Back-calculate the integral error
            self.integral_term -= error * delta_time
        else:
            self.output_value = u

        # Update the last output value
        self.last_output_value = self.output_value

        # Record state, error, y(t), and sample time values
        self.state_history.append(state_value)
        self.error_history.append(error)
        self.output_history.append(self.output_value)
        # Convert from msec to sec
        self.sample_times.append(current_time / 1000)

        # Return controller output
        return self.output_value

    def generate_dbs_signal(
        self,
        start_time,
        stop_time,
        dt,
        amplitude,
        frequency,
        pulse_width,
        offset,
        last_pulse_time_prior=0,
    ):
        return generate_monophasic_square_dbs_signal(
            start_time,
            stop_time,
            dt,
            amplitude,
            frequency,
            pulse_width,
            offset,
            last_pulse_time_prior
            )

    def set_kp(self, proportional_gain):
        """Determine how aggressively the controller reacts to the current
        error with setting Proportional Gain"""
        self.kp = proportional_gain
        self.label = "standard_PID_Controller/Kp=%f, Ti=%f, Td=%f" % (
            self.kp,
            self.ti,
            self.td,
        )

    def set_ti(self, Ti):
        """Determine how fast the controller integrates the error history by
        setting Integral Time Constant"""
        self.ti = Ti
        self.label = "standard_PID_Controller/Kp=%f, Ti=%f, Td=%f" % (
            self.kp,
            self.ti,
            self.td,
        )

    def set_td(self, Td):
        """Determine far into the future the controller predicts future errors
        setting Derivative Time Constant"""
        self.td = Td
        self.label = "standard_PID_Controller/Kp=%f, Ti=%f, Td=%f" % (
            self.kp,
            self.ti,
            self.td,
        )

    def set_setpoint(self, set_point):
        """Set target set point value"""
        self.setpoint = set_point

    def set_max_value(self, max_value):
        """Sets the upper bound for the controller output"""
        self.max_value = max_value

    def set_min_value(self, min_value):
        """Sets the lower bound for the controller output"""
        self.min_value = min_value


class IterativeFeedbackTuningPIController:
    def __init__(
        self,
        stage_length,
        setpoint=0.0,
        kp=0.0,
        ti=0.0,
        ts=0.02,
        min_value=0.0,
        max_value=1e9,
        gamma=0.005,
        lam=1e-8,
        min_kp=0,
        min_ti=0,
        fix_ti=False,
        fix_kp=False,
        r_matrix="identity",
        stage_two_mean=False,
        debug=False,
        normalise_error=True
    ):

        self.setpoint = setpoint
        self.stage_length = stage_length
        self.stage_length_samples = int(np.ceil(stage_length / ts)) + 1
        self.kp = kp
        self.ti = ti
        self.gamma = gamma
        self.lam = lam
        self.iteration_stage = -1
        self.min_kp = min_kp
        self.min_ti = min_ti
        self.r = np.identity(2)
        self.fix_ti = fix_ti
        self.fix_kp = fix_kp
        self.r_matrix = r_matrix
        self.stage_two_mean = stage_two_mean
        self.normalise_error = normalise_error

        # Set output value bounds
        self.min_value = min_value
        self.max_value = max_value

        self.debug = debug

        self.label = "iterative_feedback_tuning_PI_Controller"

        self.ts = ts
        self.current_time = 0.0  # (sec)
        self.last_time = 0.0
        self.stage_start_time = 0.0

        # Initialize controller terms
        self.integral_term = 0.0
        self.last_error = 0.0
        self.last_output_value = 0.0

        # Initialize the output value of the controller
        self.output_value = 0.0

        self._integral_term_history = []
        self._state_history = []
        self._error_history = []
        self._output_history = []
        self._sample_times = []
        self._iteration_history = []
        self._reference_history = []
        self._parameter_history = []
        self._recorded_output = np.zeros(self.stage_length_samples)
        self._mean_recorded_output = 0

    def clear(self):
        self.integral_term = 0.0
        self.last_error = 0.0

        self._integral_term_history = []
        self._state_history = []
        self._error_history = []
        self._output_history = []
        self._sample_times = []
        self._iteration_history = []
        self._parameter_history = []
        self._recorded_output = np.zeros(self.stage_length_samples)

        self.output_value = 0.0

    def dc_drho(self, s):
        kp = self.kp
        ti = self.ti
        ts = self.ts
        yout_kp = np.zeros(len(s) + 1)
        yout_ti = np.zeros(len(s) + 1)
        for i, u in enumerate(s):
            if kp == 0:
                yout_kp[i + 1] = 0
            else:
                yout_kp[i + 1] = u / kp
            if ti == 0:
                yout_ti[i + 1] = 0
            else:
                yout_ti[i + 1] = (ti ** 2 * yout_ti[i] - ts * u) / (ti ** 2 + ti * ts)

        return yout_kp[1:], yout_ti[1:]

    def compute_fitness_gradient(self):
        n_samples = self.stage_length_samples
        y1 = np.array(self._state_history[-2 * n_samples: -n_samples])
        y2 = np.array(self._state_history[-n_samples:])
        u1 = np.array(self._output_history[-2 * n_samples: -n_samples])
        u2 = np.array(self._output_history[-n_samples:])

        if self.normalise_error:
            y_tilde = (-y1 + self.setpoint) / self.setpoint
        else:
            y_tilde = -y1 + self.setpoint
        

        u_rho = u1
        dy_dkp, dy_dti = self.dc_drho(y2)
        du_dkp, du_dti = self.dc_drho(u2)
        if rank == 0 and self.debug:
            print(f"kp={self.kp}")
            print(f"y_tilde: min={y_tilde.min()}, max={y_tilde.max()}, mean={y_tilde.mean()}, sum={y_tilde.sum()}")
            print(f"dy_dkp: min={dy_dkp.min()}, max={dy_dkp.max()}, mean={dy_dkp.mean()}, sum={dy_dkp.sum()}")
            print(f"y2: min={y2.min()}, max={y2.max()}, mean={y2.mean()}, sum={y2.sum()}")
            print(f"y1: min={y1.min()}, max={y1.max()}, mean={y1.mean()}, sum={y1.sum()}")
            print(f"u1: min={u1.min()}, max={u1.max()}, mean={u1.mean()}, sum={u1.sum()}")
            print(f"du_dkp: min={du_dkp.min()}, max={du_dkp.max()}, mean={du_dkp.mean()}, sum={du_dkp.sum()}")

        dy_drho = np.vstack((dy_dkp, dy_dti))
        du_drho = np.vstack((du_dkp, du_dti))

        if self.r_matrix == "identity":
            self.r = np.identity(2)
        elif self.r_matrix == "hessian":
            r = (np.dot(dy_drho, np.transpose(dy_drho)) + self.lam * np.dot(du_drho, np.transpose(du_drho))) / self.stage_length_samples
            if self.fix_ti or self.fix_kp:
                r = np.diag(np.diag(r))  # Remove off-diagonal elements
            self.r = r

        lam = self.lam
        y_part = y_tilde * dy_drho
        u_part = u_rho * du_drho

        grad = np.sum((y_part + lam * u_part), axis=-1) / self.stage_length_samples
        return grad

    def new_controller_parameters(self):
        rho = np.array([self.kp, self.ti])
        gamma = self.gamma
        if len(self._error_history) >= 2 * self.stage_length_samples:
            grad = self.compute_fitness_gradient()
            if rank == 0:
                print(f"Gradient: {grad}'")
                print(f"R-matrix: {self.r}")
        else:
            grad = [0, 0]
            if rank == 0:
                print("Too few samples, skipping update")
        try:
            r_inv = np.linalg.inv(self.r)
        except LinAlgError:
            r_inv = np.identity(2)
        scaled_grad = np.dot(r_inv, grad)
        if rank == 0:
            print(f"R-scaled gradient: {scaled_grad}")
        new_rho = rho - gamma * scaled_grad

        if self.fix_kp:
            new_rho[0] = rho[0]
        if self.fix_ti:
            new_rho[1] = rho[1]

        if new_rho[0] < self.min_kp:
            new_rho[0] = self.min_kp
        if new_rho[1] < self.min_ti:
            new_rho[1] = self.min_ti

        return new_rho[0], new_rho[1]

    def reference_signal(self, elapsed_time):
        if self.stage_two_mean:
            return self._mean_recorded_output

        sample = int(elapsed_time / self.ts)
        if self.iteration_stage == 0 or self.iteration_stage == -1:
            return self.setpoint
        if sample >= self.stage_length_samples:
            sample = self.stage_length_samples - 1
        return self._recorded_output[sample]

    def update(self, state_value, current_time):
        self.current_time = current_time
        elapsed_time = (current_time - self.stage_start_time) / 1000
        reference = self.reference_signal(elapsed_time)
        if rank == 0:
            print(
                "Stage: %d, Elapsed time: %.3f s, Reference: %.5f"
                % (self.iteration_stage, elapsed_time, reference)
            )

        if elapsed_time >= self.stage_length:
            if (
                self.iteration_stage == 0
                and len(self._error_history) < self.stage_length_samples
            ):
                if rank == 0:
                    print("Extending stage 0 to gather more samples")
            elif (
                self.iteration_stage == 1
                and len(self._error_history) < 2 * self.stage_length_samples
            ):
                if rank == 0:
                    print("Extending stage 1 to gather more samples")
            else:
                if self.iteration_stage == 1:
                    self.kp, self.ti = self.new_controller_parameters()
                    if rank == 0:
                        print(f"New params: kp={self.kp}, ti={self.ti}")
                if self.iteration_stage == 0:
                    self._mean_recorded_output = np.mean(self._recorded_output)
                self.stage_start_time = self.current_time
                elapsed_time = 0
                self.iteration_stage = (self.iteration_stage + 1) % 2
                if rank == 0:
                    print("Stage change, now at stage %d" % self.iteration_stage)

        if self.iteration_stage == 0:
            sample = int(elapsed_time / self.ts)
            self._recorded_output[sample] = state_value
        
        if reference != 0:
            if self.normalise_error:
                error = (state_value - reference) / reference
            else:
                error = state_value - reference
        else:
            error = state_value

        self.integral_term += error * self.ts

        # Calculate u(t)
        try:
            u = self.kp * (error + ((1.0 / self.ti) * self.integral_term))
        except ZeroDivisionError:
            u = self.kp * (error + (0.0 * self.integral_term))

        # Bound the controller output
        if u > self.max_value:
            self.output_value = self.max_value
            # Back-calculate the integral error
            self.integral_term -= error * self.ts
        elif u < self.min_value:
            self.output_value = self.min_value
            # Back-calculate the integral error
            self.integral_term -= error * self.ts
        else:
            self.output_value = u

        self._integral_term_history.append(self.integral_term)
        self._state_history.append(state_value)
        self._error_history.append(error)
        self._iteration_history.append(self.iteration_stage)
        self._reference_history.append(reference)
        self._sample_times.append(current_time / 1000)
        self._output_history.append(self.output_value)
        self._parameter_history.append([self.kp, self.ti])

        return self.output_value

    def generate_dbs_signal(
        self,
        start_time,
        stop_time,
        dt,
        amplitude,
        frequency,
        pulse_width,
        offset,
        last_pulse_time_prior=0,
    ):
        return generate_monophasic_square_dbs_signal(
            start_time,
            stop_time,
            dt,
            amplitude,
            frequency,
            pulse_width,
            offset,
            last_pulse_time_prior
            )

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def setpoint(self):
        return self._setpoint

    @setpoint.setter
    def setpoint(self, value):
        self._setpoint = value

    @property
    def max_value(self):
        return self._max_value

    @max_value.setter
    def max_value(self, value):
        self._max_value = value

    @property
    def min_value(self):
        return self._min_value

    @min_value.setter
    def min_value(self, value):
        self._min_value = value

    @property
    def integral_term_history(self):
        return self._integral_term_history

    @property
    def recorded_output(self):
        return self._recorded_output

    @property
    def state_history(self):
        return self._state_history

    @property
    def error_history(self):
        return self._error_history

    @property
    def output_history(self):
        return self._output_history

    @property
    def sample_times(self):
        return self._sample_times

    @property
    def iteration_history(self):
        return self._iteration_history

    @property
    def reference_history(self):
        return self._reference_history

    @property
    def parameter_history(self):
        return self._parameter_history


#
#  Utility functions
#
def generate_monophasic_square_dbs_signal(
    start_time,
    stop_time,
    dt,
    amplitude,
    frequency,
    pulse_width,
    offset,
    last_pulse_time_prior=0,
):
    """
    Generate monophasic square-wave DBS signal

    Parameters:
        start_time (float): Stimulation start time (ms)
        stop_time (float): Stimulation stop time (ms)
        dt (float): Timestep (ms)
        amplitude (float): Stimulation amplitude (mA); <0 = cathodic, >0 = anodic
        frequency (float): Stimulation frequency (Hz)
        pulse_width (float): Width of the stimulation pulse (ms)
        offset (float): DC offset of the signal (mA)
        last_pulse_time_prior (float): Time of the last delivered pulse (ms)

    Returns:
        dbs_signal (np.array): DBS stimulation waveform
        times (np.array): sample times
        next_pulse_time (float): Time of the next stimulation pulse
        last_pulse_time (float): Time of the last stimulation pulse
    """

    times = np.round(np.arange(start_time, stop_time, dt), 2)
    tmp = np.arange(0, stop_time - start_time, dt) / 1000.0

    if frequency == 0:
        dbs_signal = np.zeros(len(tmp))
        last_pulse_time = last_pulse_time_prior
        next_pulse_time = 1e9
    else:
        # Calculate the duty cycle of the DBS signal
        isi = 1000.0 / frequency  # time is in ms
        duty_cycle = pulse_width / isi
        tt = 2.0 * np.pi * frequency * tmp
        dbs_signal = offset + 0.5 * (1.0 + signal.square(tt, duty=duty_cycle))
        dbs_signal[-1] = 0.0

        # Calculate the time for the first pulse of the next segment
        try:
            last_pulse_index = np.where(np.diff(dbs_signal) < 0)[0][-1]
            next_pulse_time = times[last_pulse_index] + isi - pulse_width

            # Track when the last pulse was
            last_pulse_time = times[last_pulse_index]

        except IndexError:
            # Catch times when signal may be flat
            last_pulse_index = len(dbs_signal) - 1
            next_pulse_time = times[last_pulse_index] + isi - pulse_width
            last_pulse_time = last_pulse_time_prior

        # Rescale amplitude
        dbs_signal *= amplitude

    return dbs_signal, times, next_pulse_time, last_pulse_time
