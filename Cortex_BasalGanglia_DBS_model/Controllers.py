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


class ZeroController:
    """Dummy controller with no stimulation"""

    def __init__(self, setpoint=0.0, Ts=0.02):

        self.SetPoint = setpoint
        self.label = "ZeroController"

        self.Ts = Ts
        self.current_time = 0.0  # (sec)
        self.last_time = 0.0

        # Initialize controller terms
        self.last_error = 0.0
        self.last_OutputValue = 0.0

        # Initialize the output value of the controller
        self.OutputValue = 0.0

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

        self.OutputValue = 0.0

    def update(self, state_value, current_time):
        """Update controller state
        """

        # Calculate Error - if SetPoint > 0.0, then normalize error with
        # respect to set point
        if self.SetPoint == 0.0:
            error = state_value - self.SetPoint
        else:
            error = (state_value - self.SetPoint)/self.SetPoint

        # Converting from msec to sec
        self.current_time = current_time / 1000.0

        # Remember last time and last error for next calculation
        self.last_time = self.current_time
        self.last_error = error

        self.OutputValue = 0

        # Update the last output value
        self.last_OutputValue = self.OutputValue

        # Record state, error, y(t), and sample time values
        self.state_history.append(state_value)
        self.error_history.append(error)
        self.output_history.append(self.OutputValue)
        # Convert from msec to sec
        self.sample_times.append(current_time / 1000)

        # Return controller output
        return self.OutputValue

    def generate_dbs_signal(self, start_time, stop_time, dt, amplitude,
                            frequency, pulse_width, offset,
                            last_pulse_time_prior=0):
        """Generate monophasic square-wave DBS signal

        Example inputs:
            start_time = 0                # ms
            stop_time = 12000            # ms
            dt = 0.01                    # ms
            amplitude = -1.0            # mA (<0 = cathodic, >0 = anodic)
            frequency = 130.0            # Hz
            pulse_width    = 0.06            # ms
            offset = 0                    # mA
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
            dbs_signal = offset + 0.5 * (1.0 + signal.square(tt,
                                                             duty=duty_cycle))
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

            # Rescale amplitude
            dbs_signal *= amplitude

        return dbs_signal, times, next_pulse_time, last_pulse_time

    def setSetPoint(self, set_point):
        """Set target set point value"""
        self.SetPoint = set_point

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


class ConstantController:
    """Constant DBS Parameter Controller Class"""

    def __init__(self, SetPoint=0.0, MinValue=0.0, MaxValue=1e9,
                 ConstantValue=0.0, Ts=0.0, units='mA'):
        # Initial Controller Values
        self.SetPoint = SetPoint
        self.MaxValue = MaxValue
        self.MinValue = MinValue
        self.ConstantValue = ConstantValue
        self.Ts = Ts                        # should be in sec as per above
        self.units = units
        self.label = ('Constant_Controller/%f%s' % (self.ConstantValue,
                                                    self.units))

        # Set output value
        self.OutputValue = 0

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

        self.OutputValue = 0.0

    def update(self, state_value, current_time):
        """Calculates biomarker for constant DBS value

            u = self.ConstantValue

        """

        # Calculate Error - if SetPoint > 0.0
        # normalize error with respect to set point
        if self.SetPoint == 0.0:
            error = state_value - self.SetPoint
        else:
            error = (state_value - self.SetPoint) / self.SetPoint

        # Bound the controller output (between MinValue - MaxValue)
        if self.ConstantValue > self.MaxValue:
            self.OutputValue = self.MaxValue
        elif self.ConstantValue < self.MinValue:
            self.OutputValue = self.MinValue
        else:
            self.OutputValue = self.ConstantValue

        # Record state, error and sample time values
        self.state_history.append(state_value)
        self.error_history.append(error)
        self.output_history.append(self.OutputValue)
        # Convert from msec to sec
        self.sample_times.append(current_time/1000)

        return self.OutputValue

    def setMaxValue(self, max_value):
        """Sets the upper bound for the controller output"""
        self.MaxValue = max_value

    def setMinValue(self, min_value):
        """Sets the lower bound for the controller output"""
        self.MinValue = min_value

    def setConstantValue(self, constant_value):
        """Sets the constant controller output"""
        self.ConstantValue = constant_value

    def setTs(self, Ts):
        """Sets the sampling rate of the controller"""
        self.Ts = Ts

    def setLabel(self, label):
        """Sets the label of the controller"""
        self.label = label

    def setSetPoint(self, set_point):
        self.SetPoint = set_point

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


class OnOffController:
    """On-Off Controller Class"""

    def __init__(self, SetPoint=0.0, MinValue=0.0, MaxValue=1e9,
                 RampDuration=0.25, Ts=0.02):
        # Initial Controller Values
        self.SetPoint = SetPoint
        self.MaxValue = MaxValue
        self.MinValue = MinValue
        # should be defined in sec, i.e. 0.25 sec
        self.RampDuration = RampDuration
        # should be in sec as per above
        self.Ts = Ts
        self.label = 'On_Off_Controller'

        # Calculate how much controller output value will change each
        # controller call
        self.OutputValueIncrement = ((self.MaxValue - self.MinValue)
                                     / math.ceil(self.RampDuration/self.Ts))

        # Initialize the output value of the controller
        self.LastOutputValue = 0
        self.OutputValue = 0

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

        self.LastOutputValue = 0.0
        self.OutputValue = 0.0

    def update(self, state_value, current_time):
        """Calculates updated controller output value for given reference feedback

            y(t) = y(t-1) + u(t)

            where:

            u(t) = MaxValue / (RampDuration/Ts) if e(t) > SetPoint
            or -MaxValue / (RampDuration/Ts) if e(t) < SetPoint

        """

        # Calculate Error - if SetPoint > 0.0
        # normalize error withrespect to set point
        if self.SetPoint == 0.0:
            error = state_value - self.SetPoint
            increment = 0.0
        else:
            error = (state_value - self.SetPoint)/self.SetPoint
            if error > 0.0:
                increment = self.OutputValueIncrement
            else:
                increment = -self.OutputValueIncrement

        # Bound the controller output (between MinValue - MaxValue)
        if self.LastOutputValue + increment > self.MaxValue:
            self.OutputValue = self.MaxValue
        elif self.LastOutputValue + increment < self.MinValue:
            self.OutputValue = self.MinValue
        else:
            self.OutputValue = self.LastOutputValue + increment

        # Record state, error and sample time values
        self.state_history.append(state_value)
        self.error_history.append(error)
        self.output_history.append(self.OutputValue)
        # Convert from msec to sec
        self.sample_times.append(current_time / 1000)

        self.LastOutputValue = self.OutputValue

        return self.OutputValue

    def setMaxValue(self, max_value):
        """Sets the upper bound for the controller output"""
        self.MaxValue = max_value
        self.OutputValueIncrement = ((self.MaxValue - self.MinValue)
                                     / (self.RampDuration / self.Ts))

    def setMinValue(self, min_value):
        """Sets the lower bound for the controller output"""
        self.MinValue = min_value
        self.OutputValueIncrement = ((self.MaxValue - self.MinValue)
                                     / (self.RampDuration / self.Ts))

    def setRampDuration(self, ramp_duration):
        """Sets how long the controller output takes to reach its max value"""
        self.RampDuration = ramp_duration
        self.OutputValueIncrement = ((self.MaxValue - self.MinValue)
                                     / (self.RampDuration / self.Ts))

    def setTs(self, Ts):
        """Sets the sampling rate of the controller"""
        self.Ts = Ts
        self.OutputValueIncrement = ((self.MaxValue - self.MinValue)
                                     / (self.RampDuration / self.Ts))

    def setLabel(self, label):
        """Sets the label of the controller"""
        self.label = label

    def setSetPoint(self, set_point):
        self.SetPoint = set_point

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

    def __init__(self, LowerThreshold=0.0, UpperThreshold=0.1, MinValue=0.0,
                 MaxValue=1e9, RampDuration=0.25, Ts=0.02):
        # Initial Controller Values
        self.UpperThreshold = UpperThreshold
        self.LowerThreshold = LowerThreshold
        self.MaxValue = MaxValue
        self.MinValue = MinValue
        # should be defined in sec, i.e. 0.25 sec
        self.RampDuration = RampDuration
        # should be in sec as per above
        self.Ts = Ts
        self.label = 'Dual_Threshold_Controller'

        # Calculate how much controller output value will change
        # each controller call
        self.OutputValueIncrement = ((self.MaxValue - self.MinValue)
                                     / math.ceil(self.RampDuration / self.Ts))

        # Initialize the output value of the controller
        self.LastOutputValue = 0.0
        self.OutputValue = 0.0

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

        self.LastOutputValue = 0.0
        self.OutputValue = 0.0

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
        if state_value > self.UpperThreshold:
            error = (state_value - self.UpperThreshold)/self.UpperThreshold
            increment = self.OutputValueIncrement
        # Decrease if below lower threshold
        elif state_value < self.LowerThreshold:
            error = (state_value - self.LowerThreshold)/self.LowerThreshold
            increment = -self.OutputValueIncrement
        # Do nothing when within upper and lower thresholds
        else:
            error = 0
            increment = 0

        # Bound the controller output (between MinValue - MaxValue)
        if self.LastOutputValue + increment > self.MaxValue:
            self.OutputValue = self.MaxValue
        elif self.LastOutputValue+increment < self.MinValue:
            self.OutputValue = self.MinValue
        else:
            self.OutputValue = self.LastOutputValue + increment

        # Record state, error and sample time values
        self.state_history.append(state_value)
        self.error_history.append(error)
        self.output_history.append(self.OutputValue)
        # Convert from msec to sec
        self.sample_times.append(current_time/1000)

        self.LastOutputValue = self.OutputValue

        return self.OutputValue

    def setUpperThreshold(self, upper_threshold):
        """Sets the upper threshold for the measured state"""
        self.UpperThreshold = upper_threshold

    def setLowerThreshold(self, lower_threshold):
        """Sets the lower threshold for the measured state"""
        self.LowerThreshold = lower_threshold

    def setMaxValue(self, max_value):
        """Sets the upper bound for the controller output"""
        self.MaxValue = max_value
        self.OutputValueIncrement = ((self.MaxValue - self.MinValue)
                                     / (self.RampDuration / self.Ts))

    def setMinValue(self, min_value):
        """Sets the lower bound for the controller output"""
        self.MinValue = min_value
        self.OutputValueIncrement = ((self.MaxValue - self.MinValue)
                                     / (self.RampDuration / self.Ts))

    def setRampDuration(self, ramp_duration):
        """Sets how long the controller output takes to reach its max value"""
        self.RampDuration = ramp_duration
        self.OutputValueIncrement = ((self.MaxValue - self.MinValue)
                                     / (self.RampDuration / self.Ts))

    def setTs(self, Ts):
        """Sets the sampling rate of the controller"""
        self.Ts = Ts
        self.OutputValueIncrement = ((self.MaxValue - self.MinValue)
                                     / (self.RampDuration / self.Ts))

    def setLabel(self, label):
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

    def __init__(self, SetPoint=0.0, Kp=0.0, Ti=0.0, Td=0.0, Ts=0.02,
                 MinValue=0.0, MaxValue=1e9):

        self.SetPoint = SetPoint
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td

        # Set output value bounds
        self.MinValue = MinValue
        self.MaxValue = MaxValue

        self.label = ("standard_PID_Controller/Kp=%f, Ti=%f, Td=%f" %
                      (self.Kp, self.Ti, self.Td))

        self.Ts = Ts
        self.current_time = 0.0  # (sec)
        self.last_time = 0.0

        # Initialize controller terms
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.last_OutputValue = 0.0

        # Initialize the output value of the controller
        self.OutputValue = 0.0

        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

    def clear(self):
        """Clears PID computations and coefficients"""

        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        self.state_history = []
        self.error_history = []
        self.output_history = []
        self.sample_times = []

        self.OutputValue = 0.0

    def update(self, state_value, current_time):
        """Calculates controller output signal for given reference feedback

            where:

            u(t) = K_p (e(t) + (1/T_i)* \\int_{0}^{t} e(t)dt + T_d {de}/{dt})

            where the error calculated is the tracking error (r(t) - y(t))

        """

        # Calculate Error - if SetPoint > 0.0, then normalize error with
        # respect to set point
        if self.SetPoint == 0.0:
            error = state_value - self.SetPoint
        else:
            error = (state_value - self.SetPoint)/self.SetPoint

        # Converting from msec to sec
        self.current_time = current_time / 1000.0
        delta_time = self.Ts
        delta_error = error - self.last_error

        self.ITerm += error * delta_time

        self.DTerm = 0.0
        if delta_time > 0:
            self.DTerm = delta_error / delta_time

        # Remember last time and last error for next calculation
        self.last_time = self.current_time
        self.last_error = error

        # Calculate u(t) - catch potential division by zero error
        try:
            u = self.Kp * (error + ((1.0/self.Ti) * self.ITerm)
                           + (self.Td * self.DTerm))
        except ZeroDivisionError:
            u = self.Kp * (error + (0.0 * self.ITerm)
                           + (self.Td * self.DTerm))

        # Bound the controller output if necessary
        # (between MinValue - MaxValue)
        if u > self.MaxValue:
            self.OutputValue = self.MaxValue
            # Back-calculate the integral error
            self.ITerm -= error * delta_time
        elif u < self.MinValue:
            self.OutputValue = self.MinValue
            # Back-calculate the integral error
            self.ITerm -= error * delta_time
        else:
            self.OutputValue = u

        # Update the last output value
        self.last_OutputValue = self.OutputValue

        # Record state, error, y(t), and sample time values
        self.state_history.append(state_value)
        self.error_history.append(error)
        self.output_history.append(self.OutputValue)
        # Convert from msec to sec
        self.sample_times.append(current_time / 1000)

        # Return controller output
        return self.OutputValue

    def generate_dbs_signal(self, start_time, stop_time, dt, amplitude,
                            frequency, pulse_width, offset,
                            last_pulse_time_prior=0):
        """Generate monophasic square-wave DBS signal

        Example inputs:
            start_time = 0                # ms
            stop_time = 12000            # ms
            dt = 0.01                    # ms
            amplitude = -1.0            # mA (<0 = cathodic, >0 = anodic)
            frequency = 130.0            # Hz
            pulse_width    = 0.06            # ms
            offset = 0                    # mA
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
            dbs_signal = offset + 0.5 * (1.0 + signal.square(tt,
                                                             duty=duty_cycle))
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

            # Rescale amplitude
            dbs_signal *= amplitude

        return dbs_signal, times, next_pulse_time, last_pulse_time

    def setKp(self, proportional_gain):
        """Determine how aggressively the controller reacts to the current
        error with setting Proportional Gain"""
        self.Kp = proportional_gain
        self.label = ("standard_PID_Controller/Kp=%f, Ti=%f, Td=%f"
                      % (self.Kp, self.Ti, self.Td))

    def setTi(self, Ti):
        """Determine how fast the controller integrates the error history by
        setting Integral Time Constant"""
        self.Ti = Ti
        self.label = ("standard_PID_Controller/Kp=%f, Ti=%f, Td=%f"
                      % (self.Kp, self.Ti, self.Td))

    def setTd(self, Td):
        """Determine far into the future the controller predicts future errors
        setting Derivative Time Constant"""
        self.Td = Td
        self.label = ("standard_PID_Controller/Kp=%f, Ti=%f, Td=%f"
                      % (self.Kp, self.Ti, self.Td))

    def setSetPoint(self, set_point):
        """Set target set point value"""
        self.SetPoint = set_point

    def setMaxValue(self, max_value):
        """Sets the upper bound for the controller output"""
        self.MaxValue = max_value

    def setMinValue(self, min_value):
        """Sets the lower bound for the controller output"""
        self.MinValue = min_value

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


class IterativeFeedbackTuningPIController():

    def __init__(self, stage_length, setpoint=0.0, kp_init=0.0, ti_init=0.0,
                 ts=0.02, min_value=0.0, max_value=1e9, step_gain=1):

        self.setpoint = setpoint
        self.stage_length = stage_length
        self.stage_length_samples = int(np.ceil(stage_length / ts)) + 1
        self.kp = kp_init
        self.ti = ti_init
        self.step_gain = step_gain
        self.iteration_stage = 1

        # Set output value bounds
        self.min_value = min_value
        self.max_value = max_value

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

    def compute_fitness_gradient(self):
        y1 = self.error_history[
            -3 * self.stage_length_samples: -2 * self.stage_length_samples]
        y2 = self.error_history[
            -2 * self.stage_length_samples: -self.stage_length_samples]
        u1 = self.output_history[
            -3 * self.stage_length_samples: -2 * self.stage_length_samples]
        u2 = self.output_history[
            -2 * self.stage_length_samples: -self.stage_length_samples]
        y_tilde = np.array(y1) - self.setpoint
        u_rho = u1
        tstart = (self.current_time / 1000) - 2 * self.stage_length
        tend = (self.current_time / 1000) - 1 * self.stage_length
        kp = self.kp
        ti = self.ti
        _, dy_dkp, _ = signal.lsim(([ti, 1], [kp * ti, kp]),
                                   y2,
                                   np.linspace(tstart, tend,
                                               self.stage_length_samples))
        print(dy_dkp)
        _, dy_dti, _ = signal.lsim(([-1], [ti ** 2, ti]),
                                   y2,
                                   np.linspace(tstart, tend,
                                               self.stage_length_samples))

        _, du_dkp, _ = signal.lsim(([ti, 1], [kp * ti, kp]),
                                   u2,
                                   np.linspace(tstart, tend,
                                               self.stage_length_samples))
        _, du_dti, _ = signal.lsim(([-1], [ti ** 2, ti]),
                                   u2,
                                   np.linspace(tstart, tend,
                                               self.stage_length_samples))

        du_drho = np.vstack((du_dkp, du_dti))
        dy_drho = np.vstack((dy_dkp, dy_dti))

        lam = 1
        y_part = y_tilde * dy_drho
        u_part = u_rho * du_drho

        grad = (np.sum((y_part + lam * u_part), axis=1) /
                self.stage_length_samples)
        return grad

    def new_controller_parameters(self):
        rho = np.array([self.kp, self.ti])
        # TODO: Make gamma and r controller parameters
        gamma = 1.0
        r = np.identity(2)
        grad = self.compute_fitness_gradient()
        new_rho = rho - gamma * np.dot(r, grad)
        return new_rho[0], new_rho[1]

    def reference_signal(self, elapsed_time):
        sample = int(elapsed_time / self.ts)
        if self.iteration_stage == 0:
            return self.setpoint
        if sample > self.stage_length_samples:
            sample = self.stage_length_samples - 1
        return self._recorded_output[sample]

    def update(self, state_value, current_time):
        self.current_time = current_time
        elapsed_time = (current_time - self.stage_start_time) / 1000
        setpoint = self.reference_signal(elapsed_time)
        print('Stage: %d, Elapsed time: %.3f s, Reference: %.5f' %
              (self.iteration_stage, elapsed_time, setpoint))
        if self.iteration_stage == 0:
            sample = int(elapsed_time / self.ts)
            self._recorded_output[sample] = state_value
        if setpoint != 0:
            error = (state_value - setpoint) / setpoint
        else:
            error = state_value

        if elapsed_time >= self.stage_length:
            if self.iteration_stage == 1:
                self.kp, self.ti = self.new_controller_parameters()
                print(f'New params: kp={self.kp}, ti={self.ti}')
            self.stage_start_time = self.current_time
            self.iteration_stage = (self.iteration_stage + 1) % 2
            print('Stage change, now at stage %d' % self.iteration_stage)

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
        self._reference_history.append(setpoint)
        self._sample_times.append(current_time / 1000)
        self._output_history.append(self.output_value)
        self._parameter_history.append([self.kp, self.ti])

        return self.output_value

    def generate_dbs_signal(self, start_time, stop_time, dt, amplitude,
                            frequency, pulse_width, offset,
                            last_pulse_time_prior=0):
        """Generate monophasic square-wave DBS signal

        Example inputs:
            start_time = 0                # ms
            stop_time = 12000            # ms
            dt = 0.01                    # ms
            amplitude = -1.0            # mA (<0 = cathodic, >0 = anodic)
            frequency = 130.0            # Hz
            pulse_width    = 0.06            # ms
            offset = 0                    # mA
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
            dbs_signal = offset + 0.5 * (1.0 + signal.square(tt,
                                                             duty=duty_cycle))
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

            # Rescale amplitude
            dbs_signal *= amplitude

        return dbs_signal, times, next_pulse_time, last_pulse_time

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
