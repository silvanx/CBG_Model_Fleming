#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description:
    Cortico-Basal Ganglia Network Model implemented in PyNN using the NEURON simulator.
    This version of the model runs to a steady state and implements either DBS
    ampltiude or frequency modulation controllers, where the beta ARV from the STN LFP
    is calculated at each controller call and used to update the amplitude/frequency of
    the DBS waveform that is applied to the network.

    Full documentation of the model and controllers used is given in:
    https://www.frontiersin.org/articles/10.3389/fnins.2020.00166/

Original author: John Fleming, john.fleming@ucdconnect.ie
"""
import os
from pathlib import Path

# Change working directory so that imports works (save old)
oldpwd = Path(os.getcwd()).resolve()
newpwd = Path(__file__).resolve().parent
os.chdir(newpwd)

# No GUI please
opts = os.environ.get("NEURON_MODULE_OPTIONS", "")
if "nogui" not in opts:
    os.environ["NEURON_MODULE_OPTIONS"] = opts + " -nogui"

from mpi4py import MPI
import neuron
from pyNN.neuron import setup, run_until, end, simulator
from pyNN.parameters import Sequence
from Controllers import (
    ZeroController,
    StandardPIDController,
    IterativeFeedbackTuningPIController,
    ConstantController,
)
import neo.io
import quantities as pq
import numpy as np
import math
import argparse
from utils import make_beta_cheby1_filter, calculate_avg_beta_power
from model import create_network, load_network, electrode_distance
from config import Config, get_controller_kwargs

# Import global variables for GPe DBS
import Global_Variables as GV

h = neuron.h
comm = MPI.COMM_WORLD

if __name__ == "__main__":
    os.chdir(oldpwd)
    parser = argparse.ArgumentParser(description="CBG Model")
    parser.add_argument("config_file", nargs="?", help="yaml configuration file")
    parser.add_argument(
        "-o", "--output-dir", default="RESULTS", help="output directory name"
    )
    args, unknown = parser.parse_known_args()

    config_file = Path(args.config_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    c = Config(args.config_file)
    os.chdir(newpwd)

    simulation_runtime = c.RunTime
    controller_type = c.Controller
    rng_seed = c.RandomSeed
    timestep = c.TimeStep
    steady_state_duration = c.SteadyStateDuration
    save_stn_voltage = c.save_stn_voltage
    beta_burst_modulation_scale = c.beta_burst_modulation_scale
    ctx_dc_offset = c.ctx_dc_offset
    Pop_size = c.Pop_size
    create_new_network = c.create_new_network
    controller_sampling_time = 1000 * c.ts
    ctx_slow_modulation_amplitude = c.ctx_slow_modulation_amplitude
    ctx_slow_modulation_step_count = c.ctx_slow_modulation_step_count

    sim_total_time = (
        steady_state_duration + simulation_runtime + timestep
    )  # Total simulation time
    rec_sampling_interval = 0.5  # Signals are sampled every 0.5 ms

    # Setup simulation
    rank = setup(timestep=timestep, rngseed=rng_seed)
    simulator.load_mechanisms("neuron_mechanisms")

    if rank == 0:
        print("\n------ Configuration ------")
        print(c, "\n")

    # Make beta band filter centred on 25Hz (cutoff frequencies are 21-29 Hz)
    # for biomarker estimation
    fs = 1000.0 / rec_sampling_interval
    beta_b, beta_a = make_beta_cheby1_filter(fs=fs, n=4, rp=0.5, low=21, high=29)

    # Use CVode to calculate i_membrane_ for fast LFP calculation
    cvode = h.CVode()
    cvode.active(0)

    # Second spatial derivative (the segment current) for the collateral
    cvode.use_fast_imem(1)

    # Set initial values for cell membrane voltages
    v_init = -68

    if not create_new_network:
        if rank == 0:
            print("Loading network...")
        (
            Pop_size,
            striatal_spike_times,
            Cortical_Pop,
            Interneuron_Pop,
            STN_Pop,
            GPe_Pop,
            GPi_Pop,
            Striatal_Pop,
            Thalamic_Pop,
            prj_CorticalAxon_Interneuron,
            prj_Interneuron_CorticalSoma,
            prj_CorticalSTN,
            prj_STNGPe,
            prj_GPeGPe,
            prj_GPeSTN,
            prj_StriatalGPe,
            prj_STNGPi,
            prj_GPeGPi,
            prj_GPiThalamic,
            prj_ThalamicCortical,
            prj_CorticalThalamic,
            GPe_stimulation_order,
        ) = load_network(
            steady_state_duration,
            sim_total_time,
            simulation_runtime,
            v_init,
            rng_seed,
            beta_burst_modulation_scale,
            ctx_dc_offset,
            ctx_slow_modulation_amplitude,
            ctx_slow_modulation_step_count
        )
        if rank == 0:
            print("Network loaded.")
    else:
        if rank == 0:
            print(f"Creating network ({Pop_size} cells per population)...")
        (
            striatal_spike_times,
            Cortical_Pop,
            Interneuron_Pop,
            STN_Pop,
            GPe_Pop,
            GPi_Pop,
            Striatal_Pop,
            Thalamic_Pop,
            prj_CorticalAxon_Interneuron,
            prj_Interneuron_CorticalSoma,
            prj_CorticalSTN,
            prj_STNGPe,
            prj_GPeGPe,
            prj_GPeSTN,
            prj_StriatalGPe,
            prj_STNGPi,
            prj_GPeGPi,
            prj_GPiThalamic,
            prj_ThalamicCortical,
            prj_CorticalThalamic,
            GPe_stimulation_order,
        ) = create_network(
            Pop_size,
            steady_state_duration,
            sim_total_time,
            simulation_runtime,
            v_init,
            rng_seed,
            beta_burst_modulation_scale,
            ctx_dc_offset,
            ctx_slow_modulation_amplitude,
            ctx_slow_modulation_step_count,
        )
        if rank == 0:
            print("Network created")

    # Define state variables to record from each population
    Cortical_Pop.record("soma(0.5).v", sampling_interval=rec_sampling_interval)
    Cortical_Pop.record("collateral(0.5).v", sampling_interval=rec_sampling_interval)
    Interneuron_Pop.record("soma(0.5).v", sampling_interval=rec_sampling_interval)
    STN_Pop.record("soma(0.5).v", sampling_interval=rec_sampling_interval)
    STN_Pop.record("AMPA.i", sampling_interval=rec_sampling_interval)
    STN_Pop.record("GABAa.i", sampling_interval=rec_sampling_interval)
    Striatal_Pop.record("spikes")
    GPe_Pop.record("soma(0.5).v", sampling_interval=rec_sampling_interval)
    GPi_Pop.record("soma(0.5).v", sampling_interval=rec_sampling_interval)
    Thalamic_Pop.record("soma(0.5).v", sampling_interval=rec_sampling_interval)

    # Assign Positions for recording and stimulating electrode point sources
    recording_electrode_1_position = np.array([0, -1500, 250])
    recording_electrode_2_position = np.array([0, 1500, 250])
    stimulating_electrode_position = np.array([0, 0, 250])

    (
        STN_recording_electrode_1_distances,
        STN_recording_electrode_2_distances,
        Cortical_Collateral_stimulating_electrode_distances,
    ) = electrode_distance(
        recording_electrode_1_position,
        recording_electrode_2_position,
        STN_Pop,
        stimulating_electrode_position,
        Cortical_Pop,
    )

    # Conductivity and resistivity values for homogenous, isotropic medium
    sigma = 0.27  # Latikka et al. 2001 - Conductivity of Brain tissue S/m
    # rho needs units of ohm cm for xtra mechanism (S/m -> S/cm)
    rho = 1 / (sigma * 1e-2)

    # Calculate transfer resistances for each collateral segment for xtra
    # units are Mohms
    collateral_rx = (
        0.01
        * (rho / (4 * math.pi))
        * (1 / Cortical_Collateral_stimulating_electrode_distances)
    )

    # Convert ndarray to array of Sequence objects - needed to set cortical
    # collateral transfer resistances
    collateral_rx_seq = np.ndarray(
        shape=(1, Cortical_Pop.local_size), dtype=Sequence
    ).flatten()
    for ii in range(0, Cortical_Pop.local_size):
        collateral_rx_seq[ii] = Sequence(collateral_rx[ii, :].flatten())

    # Assign transfer resistances values to collaterals
    for ii, cell in enumerate(Cortical_Pop):
        cell.collateral_rx = collateral_rx_seq[ii]

    # Create times for when the DBS controller will be called
    # Window length for filtering biomarker
    controller_window_length = c.controller_window_length  # ms
    controller_window_length_no_samples = int(
        controller_window_length / rec_sampling_interval
    )

    # Window Tail length - removed post filtering, prior to
    # biomarker calculation
    controller_window_tail_length = c.controller_window_tail_length  # ms
    controller_window_tail_length_no_samples = int(
        controller_window_tail_length / rec_sampling_interval
    )
    if controller_window_tail_length_no_samples > controller_window_length_no_samples * 0.3:
        print("Controller window tail length can't be longer than 1/3 of the controller window length! Resizing...")
        controller_window_tail_length_no_samples = int(controller_window_length_no_samples * 0.3)
        print(f"New controller window tail_length: {controller_window_tail_length_no_samples}")

    controller_start = (
        steady_state_duration + controller_window_length + controller_sampling_time
    )
    controller_call_times = np.arange(
        controller_start, sim_total_time, controller_sampling_time
    )

    if len(controller_call_times) == 0:
        controller_call_times = np.array([controller_start])

    # Initialize the Controller being used:
    # Controller sampling period, Ts, is in sec
    if controller_type == "ZERO":
        Controller = ZeroController
    elif controller_type == "PID":
        Controller = StandardPIDController
    elif controller_type == "IFT":
        Controller = IterativeFeedbackTuningPIController
    elif controller_type == "OPEN":
        Controller = ConstantController
    else:
        raise RuntimeError("Bad choice of Controller")

    controller_kwargs = get_controller_kwargs(c)
    controller = Controller(**controller_kwargs)

    simulation_output_dir = output_dir
    if rank == 0:
        print(f"Output directory: {simulation_output_dir}")
        simulation_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a square wave which represents the DBS signal
    # Needs to be initialized to zero when unused to prevent
    # open-circuit of cortical collateral extracellular mechanism
    if c.Modulation == "frequency":
        last_pulse_time_prior = steady_state_duration
    else:
        last_pulse_time_prior = 0
    (
        DBS_Signal,
        DBS_times,
        next_DBS_pulse_time,
        last_DBS_pulse_time,
    ) = controller.generate_dbs_signal(
        start_time=steady_state_duration + 10 + simulator.state.dt,
        stop_time=sim_total_time,
        last_pulse_time_prior=last_pulse_time_prior,
        dt=simulator.state.dt,
        amplitude=-1.0,
        frequency=130.0,
        pulse_width=0.06,
        offset=0,
    )

    DBS_Signal = np.hstack((np.array([0, 0]), DBS_Signal))
    DBS_times = np.hstack((np.array([0, steady_state_duration + 10]), DBS_times))

    # Get DBS time indexes which corresponds to controller call times
    controller_DBS_indices = []
    for call_time in controller_call_times:
        indices = np.where(DBS_times == call_time)[0]
        if len(indices) > 0:
            controller_DBS_indices.extend([indices[0]])

    # Set first portion of DBS signal (Up to first controller call after
    # steady state) to zero amplitude
    DBS_Signal[0:] = 0
    next_DBS_pulse_time = controller_call_times[0]

    DBS_Signal_neuron = h.Vector(DBS_Signal)
    DBS_times_neuron = h.Vector(DBS_times)

    # Play DBS signal to global variable is_xtra
    DBS_Signal_neuron.play(h._ref_is_xtra, DBS_times_neuron, 1)

    # Get DBS_Signal_neuron as a numpy array for easy updating
    updated_DBS_signal = DBS_Signal_neuron.as_numpy()

    # Initialize tracking the frequencies calculated by the controller
    last_freq_calculated = 0
    last_DBS_pulse_time = steady_state_duration

    # GPe DBS current stimulations - precalculated for % of collaterals
    # entrained for varying DBS amplitude
    interp_DBS_amplitudes = np.array(
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 2.25, 2.50, 3, 4, 5]
    )
    interp_collaterals_entrained = np.array(
        [0, 0, 0, 1, 4, 8, 19, 30, 43, 59, 82, 100, 100, 100]
    )

    if c.Modulation == "frequency":
        last_pulse_time_prior = steady_state_duration
    else:
        last_pulse_time_prior = 0

    # Make new GPe DBS vector for each GPe neuron - each GPe neuron needs a
    # pointer to its own DBS signal
    GPe_DBS_Signal_neuron = []
    GPe_DBS_times_neuron = []
    updated_GPe_DBS_signal = []
    for i in range(0, Cortical_Pop.local_size):
        (
            GPe_DBS_Signal,
            GPe_DBS_times,
            GPe_next_DBS_pulse_time,
            GPe_last_DBS_pulse_time,
        ) = controller.generate_dbs_signal(
            start_time=steady_state_duration + 10 + simulator.state.dt,
            stop_time=sim_total_time,
            last_pulse_time_prior=last_pulse_time_prior,
            dt=simulator.state.dt,
            amplitude=100.0,
            frequency=130.0,
            pulse_width=0.06,
            offset=0,
        )

        GPe_DBS_Signal = np.hstack((np.array([0, 0]), GPe_DBS_Signal))
        GPe_DBS_times = np.hstack(
            (np.array([0, steady_state_duration + 10]), GPe_DBS_times)
        )

        # Set the GPe DBS signals to zero amplitude
        GPe_DBS_Signal[0:] = 0
        GPe_next_DBS_pulse_time = controller_call_times[0]

        # Neuron vector of GPe DBS signals
        GPe_DBS_Signal_neuron.append(h.Vector(GPe_DBS_Signal))
        GPe_DBS_times_neuron.append(h.Vector(GPe_DBS_times))

        # Play the stimulation into each GPe neuron
        GPe_DBS_Signal_neuron[i].play(
            GV.GPe_stimulation_iclamps[i]._ref_amp, GPe_DBS_times_neuron[i], 1
        )

        # Hold a reference to the signal as a numpy array, and append to list
        # of GPe stimulation signals
        updated_GPe_DBS_signal.append(GPe_DBS_Signal_neuron[i].as_numpy())

    # Initialise STN LFP list
    STN_LFP = []
    STN_LFP_AMPA = []
    STN_LFP_GABAa = []

    # Variables for writing simulation data
    last_write_time = steady_state_duration

    if rank == 0:
        print(
            f"\n---> Running simulation to steady state ({steady_state_duration} ms) ..."
        )
    # Load the steady state
    run_until(steady_state_duration + simulator.state.dt, run_from_steady_state=False)
    if rank == 0:
        print("Steady state finished.")
        print(
            "\n---> Running simulation for %.0f ms after steady state (%.0f ms) with %s control"
            % (simulation_runtime, steady_state_duration, controller_type)
        )

    # Reload striatal spike times after loading the steady state
    Striatal_Pop.set(spike_times=striatal_spike_times[:, 0])

    # For loop to integrate the model up to each controller call
    for call_index, call_time in enumerate(controller_call_times):
        # Integrate model to controller_call_time
        run_until(call_time - simulator.state.dt)

        if rank == 0:
            print("Controller Called at t: %.2f" % simulator.state.t)

        # Calculate the LFP and biomarkers, etc.
        STN_AMPA_i = np.array(
            STN_Pop.get_data("AMPA.i", gather=False).segments[0].analogsignals[0]
        )
        STN_GABAa_i = np.array(
            STN_Pop.get_data("GABAa.i", gather=False).segments[0].analogsignals[0]
        )
        STN_Syn_i = STN_AMPA_i + STN_GABAa_i

        # STN LFP Calculation - Syn_i is in units of nA -> LFP units are mV
        STN_LFP_1 = (
            (1 / (4 * math.pi * sigma))
            * np.sum(
                (1 / (STN_recording_electrode_1_distances * 1e-6))
                * STN_Syn_i.transpose(),
                axis=0,
            )
            * 1e-6
        )
        STN_LFP_2 = (
            (1 / (4 * math.pi * sigma))
            * np.sum(
                (1 / (STN_recording_electrode_2_distances * 1e-6))
                * STN_Syn_i.transpose(),
                axis=0,
            )
            * 1e-6
        )
        STN_LFP = np.hstack(
            (STN_LFP, comm.allreduce(STN_LFP_1 - STN_LFP_2, op=MPI.SUM))
        )

        # STN LFP AMPA and GABAa Contributions
        STN_LFP_AMPA_1 = (
            (1 / (4 * math.pi * sigma))
            * np.sum(
                (1 / (STN_recording_electrode_1_distances * 1e-6))
                * STN_AMPA_i.transpose(),
                axis=0,
            )
            * 1e-6
        )
        STN_LFP_AMPA_2 = (
            (1 / (4 * math.pi * sigma))
            * np.sum(
                (1 / (STN_recording_electrode_2_distances * 1e-6))
                * STN_AMPA_i.transpose(),
                axis=0,
            )
            * 1e-6
        )
        STN_LFP_AMPA = np.hstack(
            (STN_LFP_AMPA, comm.allreduce(STN_LFP_AMPA_1 - STN_LFP_AMPA_2, op=MPI.SUM))
        )

        STN_LFP_GABAa_1 = (
            (1 / (4 * math.pi * sigma))
            * np.sum(
                (1 / (STN_recording_electrode_1_distances * 1e-6))
                * STN_GABAa_i.transpose(),
                axis=0,
            )
            * 1e-6
        )
        STN_LFP_GABAa_2 = (
            (1 / (4 * math.pi * sigma))
            * np.sum(
                (1 / (STN_recording_electrode_2_distances * 1e-6))
                * STN_GABAa_i.transpose(),
                axis=0,
            )
            * 1e-6
        )
        STN_LFP_GABAa = np.hstack(
            (
                STN_LFP_GABAa,
                comm.allreduce(STN_LFP_GABAa_1 - STN_LFP_GABAa_2, op=MPI.SUM),
            )
        )

        # Biomarker Calculation:
        lfp_beta_average_value = calculate_avg_beta_power(
            lfp_signal=STN_LFP[-controller_window_length_no_samples:],
            tail_length=controller_window_tail_length_no_samples,
            beta_b=beta_b,
            beta_a=beta_a,
        )

        if rank == 0:
            print("Beta Average: %f" % lfp_beta_average_value)

        if c.Modulation == "frequency":
            # Calculate the updated DBS Frequency
            DBS_amp = 1.5
            DBS_freq = controller.update(
                state_value=lfp_beta_average_value, current_time=simulator.state.t
            )
        else:
            # Calculate the updated DBS amplitude
            DBS_amp = controller.update(
                state_value=lfp_beta_average_value, current_time=simulator.state.t
            )
            DBS_freq = 130.0

        # Update the DBS Signal
        if call_index + 1 < len(controller_call_times):

            if c.Modulation == "frequency":
                last_pulse_time_prior = last_DBS_pulse_time
                # Check if the frequency needs to change before the last time that was calculated
                if DBS_freq != last_freq_calculated:
                    if DBS_freq == 0.0:  # Check if DBS wants to turn off
                        next_DBS_pulse_time = 1e9
                    else:  # Calculate new next pulse time if DBS is on
                        T = (1.0 / DBS_freq) * 1e3
                        next_DBS_pulse_time = last_DBS_pulse_time + T - 0.06

                        # Need to check for situation when new DBS time is less than the current time
                        if next_DBS_pulse_time <= simulator.state.t:
                            next_DBS_pulse_time = simulator.state.t
            else:
                last_pulse_time_prior = 0

            # Calculate new DBS segment from the next DBS pulse time
            if next_DBS_pulse_time < controller_call_times[call_index + 1]:

                GPe_next_DBS_pulse_time = next_DBS_pulse_time

                # DBS Cortical Collateral Stimulation
                (
                    new_DBS_Signal_Segment,
                    new_DBS_times_Segment,
                    next_DBS_pulse_time,
                    last_DBS_pulse_time,
                ) = controller.generate_dbs_signal(
                    start_time=next_DBS_pulse_time,
                    stop_time=controller_call_times[call_index + 1],
                    last_pulse_time_prior=last_pulse_time_prior,
                    dt=simulator.state.dt,
                    amplitude=-DBS_amp,
                    frequency=DBS_freq,
                    pulse_width=0.06,
                    offset=0,
                )

                # Update DBS segment - replace original DBS array values with
                # updated ones
                indices = np.where(DBS_times == new_DBS_times_Segment[0])[0]
                if len(indices) > 0:
                    window_start_index = indices[0]
                else:
                    window_start_index = 0
                new_window_sample_length = len(new_DBS_Signal_Segment)
                window_end_index = window_start_index + new_window_sample_length
                updated_DBS_signal[
                    window_start_index:window_end_index
                ] = new_DBS_Signal_Segment

                # DBS GPe neuron stimulation
                num_GPe_Neurons_entrained = int(
                    np.interp(
                        DBS_amp, interp_DBS_amplitudes, interp_collaterals_entrained
                    )
                )

                # Make copy of current DBS segment and rescale for GPe neuron
                # stimulation
                GPe_DBS_Segment = new_DBS_Signal_Segment.copy()
                GPe_DBS_Segment *= -1
                GPe_DBS_Segment[GPe_DBS_Segment > 0] = 100

                # Stimulate the entrained GPe neurons
                for i in np.arange(0, num_GPe_Neurons_entrained):
                    cellid = Cortical_Pop[GPe_stimulation_order[i]]
                    if Cortical_Pop.is_local(cellid):
                        index = Cortical_Pop.id_to_local_index(cellid)
                        updated_GPe_DBS_signal[index][
                            window_start_index:window_end_index
                        ] = GPe_DBS_Segment

                # Remember the last frequency that was calculated
                last_freq_calculated = DBS_freq

            else:
                pass

        # Write population data to file
        if save_stn_voltage:
            write_index = "{:.0f}_".format(call_index)
            suffix = "_{:.0f}ms-{:.0f}ms".format(
                last_write_time, simulator.state.t)
            fname = write_index + "STN_Soma_v" + suffix + ".mat"
            STN_Pop.write_data(
                str(simulation_output_dir / "STN_POP" / fname),
                "soma(0.5).v",
                clear=True
            )
        else:
            STN_Pop.get_data("soma(0.5).v", clear=True)

        last_write_time = simulator.state.t

    # Write population membrane voltage data to file
    if c.save_ctx_voltage:
        Cortical_Pop.write_data(str(simulation_output_dir / "Cortical_Pop/Cortical_Collateral_v.mat"), 'collateral(0.5).v', clear=False)
        Cortical_Pop.write_data(str(simulation_output_dir / "Cortical_Pop/Cortical_Soma_v.mat"), 'soma(0.5).v', clear=True)
    # Interneuron_Pop.write_data(str(simulation_output_dir / "Interneuron_Pop/Interneuron_Soma_v.mat"), 'soma(0.5).v', clear=True)
    # GPe_Pop.write_data(str(simulation_output_dir / "GPe_Pop/GPe_Soma_v.mat", 'soma(0.5).v'), clear=True)
    # GPi_Pop.write_data(str(simulation_output_dir / "GPi_Pop/GPi_Soma_v.mat", 'soma(0.5).v'), clear=True)
    # Thalamic_Pop.write_data(str(simulation_output_dir / "Thalamic_Pop/Thalamic_Soma_v.mat"), 'soma(0.5).v', clear=True)

    # Write controller values to csv files
    controller_measured_beta_values = np.asarray(controller.state_history)
    controller_measured_error_values = np.asarray(controller.error_history)
    controller_output_values = np.asarray(controller.output_history)
    controller_sample_times = np.asarray(controller.sample_times)
    try:
        controller_reference_history = np.asarray(controller.reference_history)
    except AttributeError:
        controller_reference_history = None
    try:
        controller_iteration_history = np.asarray(controller.iteration_history)
    except AttributeError:
        controller_iteration_history = None
    try:
        controller_parameter_history = np.asarray(controller.parameter_history)
    except AttributeError:
        controller_parameter_history = None
    try:
        controller_integral_term_history = np.asarray(controller.integral_term_history)
    except AttributeError:
        controller_integral_term_history = None

    if rank == 0:
        np.savetxt(
            simulation_output_dir / "controller_beta_values.csv",
            controller_measured_beta_values,
            delimiter=",",
        )
        np.savetxt(
            simulation_output_dir / "controller_error_values.csv",
            controller_measured_error_values,
            delimiter=",",
        )
        np.savetxt(
            simulation_output_dir / "controller_values.csv",
            controller_output_values,
            delimiter=",",
        )
        np.savetxt(
            simulation_output_dir / "controller_sample_times.csv",
            controller_sample_times,
            delimiter=",",
        )
        if controller_iteration_history is not None:
            np.savetxt(
                simulation_output_dir / "controller_iteration_values.csv",
                controller_iteration_history,
                delimiter=",",
            )
        if controller_reference_history is not None:
            np.savetxt(
                simulation_output_dir / "controller_reference_values.csv",
                controller_reference_history,
                delimiter=",",
            )
        if controller_parameter_history is not None:
            np.savetxt(
                simulation_output_dir / "controller_parameter_values.csv",
                controller_parameter_history,
                delimiter=",",
            )
        if controller_integral_term_history is not None:
            np.savetxt(
                simulation_output_dir / "controller_integral_term_values.csv",
                controller_integral_term_history,
                delimiter=",",
            )

    # Write the STN LFP to .mat file
    STN_LFP_Block = neo.Block(name="STN_LFP")
    STN_LFP_seg = neo.Segment(name="segment_0")
    STN_LFP_Block.segments.append(STN_LFP_seg)
    STN_LFP_signal = neo.AnalogSignal(
        STN_LFP,
        units="mV",
        t_start=0 * pq.ms,
        sampling_rate=pq.Quantity(1.0 / rec_sampling_interval, "1/ms"),
    )
    STN_LFP_seg.analogsignals.append(STN_LFP_signal)

    w = neo.io.NeoMatlabIO(filename=str(simulation_output_dir / "STN_LFP.mat"))
    w.write_block(STN_LFP_Block)

    # # Write LFP AMPA and GABAa components to file
    # STN_LFP_AMPA_Block = neo.Block(name='STN_LFP_AMPA')
    # STN_LFP_AMPA_seg = neo.Segment(name='segment_0')
    # STN_LFP_AMPA_Block.segments.append(STN_LFP_AMPA_seg)
    # STN_LFP_AMPA_signal = neo.AnalogSignal(STN_LFP_AMPA, units='mV', t_start=0*pq.ms, sampling_rate=pq.Quantity(1.0 / rec_sampling_interval, '1/ms'))
    # STN_LFP_AMPA_seg.analogsignals.append(STN_LFP_AMPA_signal)
    # w = neo.io.NeoMatlabIO(filename=str(simulation_output_dir / "STN_LFP_AMPA.mat"))
    # w.write_block(STN_LFP_AMPA_Block)

    # STN_LFP_GABAa_Block = neo.Block(name='STN_LFP_GABAa')
    # STN_LFP_GABAa_seg = neo.Segment(name='segment_0')
    # STN_LFP_GABAa_Block.segments.append(STN_LFP_GABAa_seg)
    # STN_LFP_GABAa_signal = neo.AnalogSignal(STN_LFP_GABAa, units='mV', t_start=0*pq.ms, sampling_rate=pq.Quantity(1.0 / rec_sampling_interval, '1/ms'))
    # STN_LFP_GABAa_seg.analogsignals.append(STN_LFP_GABAa_signal)
    # w = neo.io.NeoMatlabIO(filename=str(simulation_output_dir / "STN_LFP_GABAa.mat"))
    # w.write_block(STN_LFP_GABAa_Block)

    # Write the DBS Signal to .mat file
    # DBS Amplitude
    DBS_Block = neo.Block(name="DBS_Signal")
    DBS_Signal_seg = neo.Segment(name="segment_0")
    DBS_Block.segments.append(DBS_Signal_seg)
    DBS_signal = neo.AnalogSignal(
        DBS_Signal_neuron,
        units="mA",
        t_start=0 * pq.ms,
        sampling_rate=pq.Quantity(1.0 / simulator.state.dt, "1/ms"),
    )
    DBS_Signal_seg.analogsignals.append(DBS_signal)
    DBS_times = neo.AnalogSignal(
        DBS_times_neuron,
        units="ms",
        t_start=DBS_times_neuron * pq.ms,
        sampling_rate=pq.Quantity(1.0 / simulator.state.dt, "1/ms"),
    )
    DBS_Signal_seg.analogsignals.append(DBS_times)

    w = neo.io.NeoMatlabIO(filename=str(simulation_output_dir / "DBS_Signal.mat"))
    w.write_block(DBS_Block)

    if rank == 0:
        print("Simulation Done!")

    end()
