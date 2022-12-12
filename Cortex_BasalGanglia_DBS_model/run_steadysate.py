#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed April 03 14:27:26 2019

Description: Cortico-Basal Ganglia Network Model implemented in PyNN using the
            simulator Neuron. This version of the model runs an initial run of
            the model integrating the model to the steady state. The steady
            state can then be loaded in subsequent simulations to test DBS
            controllers. Full documentation of the model and controllers used
            is given in:

            https://www.frontiersin.org/articles/10.3389/fnins.2020.00166/

@author: John Fleming, john.fleming@ucdconnect.ie
"""
import os

# No GUI please
opts = os.environ.get("NEURON_MODULE_OPTIONS", "")
if "nogui" not in opts:
    os.environ["NEURON_MODULE_OPTIONS"] = opts + " -nogui"

from mpi4py import MPI
import neuron
from pyNN.neuron import setup, run_to_steady_state, end, simulator
from pyNN.parameters import Sequence
import numpy as np
import math
import neo
import quantities as pq
from utils import make_beta_cheby1_filter
from model import load_network, electrode_distance

# Import global variables for GPe DBS
import Global_Variables as GV


h = neuron.h
comm = MPI.COMM_WORLD

if __name__ == "__main__":
    rng_seed = 3695
    timestep = 0.01
    save_sim_data = False
    # Setup simulation
    rank = setup(timestep=timestep, rngseed=rng_seed)

    if rank == 0:
        print("\nSetting up simulation...")
    steady_state_duration = 6000.0  # Duration of simulation steady state
    simulation_duration = steady_state_duration  # Total simulation time
    rec_sampling_interval = 0.5  # Fs = 2000Hz

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
        simulation_duration,
        simulation_duration,
        v_init,
        rng_seed=rng_seed,
    )

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

    # Variables for writing simulation data
    last_write_time = 0

    DBS_Signal_neuron = h.Vector([0, 0])
    DBS_times_neuron = h.Vector([0, steady_state_duration + 10])

    # Play DBS signal to global variable is_xtra
    DBS_Signal_neuron.play(h._ref_is_xtra, DBS_times_neuron, 1)

    # Make new GPe DBS vector for each GPe neuron - each GPe neuron needs
    # a pointer to it's own DBS signal
    GPe_DBS_Signal_neuron = []
    GPe_DBS_times_neuron = []
    updated_GPe_DBS_signal = []
    for i in range(0, Cortical_Pop.local_size):

        # Neuron vector of GPe DBS signals
        GPe_DBS_Signal_neuron.append(h.Vector([0, 0]))
        GPe_DBS_times_neuron.append(h.Vector([0, steady_state_duration + 10]))

        # Play the stimulation into eacb GPe neuron
        GPe_DBS_Signal_neuron[i].play(
            GV.GPe_stimulation_iclamps[i]._ref_amp, GPe_DBS_times_neuron[i], 1
        )

    # Initialise STN LFP list
    STN_LFP = []
    STN_LFP_AMPA = []
    STN_LFP_GABAa = []

    # Run the model to the steady state
    if rank == 0:
        print("Running model to steady state...")

    run_to_steady_state(steady_state_duration)

    if rank == 0:
        print("Done.")

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
            (1 / (STN_recording_electrode_1_distances * 1e-6)) * STN_Syn_i.transpose(),
            axis=0,
        )
        * 1e-6
    )
    STN_LFP_2 = (
        (1 / (4 * math.pi * sigma))
        * np.sum(
            (1 / (STN_recording_electrode_2_distances * 1e-6)) * STN_Syn_i.transpose(),
            axis=0,
        )
        * 1e-6
    )
    STN_LFP = np.hstack((STN_LFP, comm.allreduce(STN_LFP_1 - STN_LFP_2, op=MPI.SUM)))

    # STN LFP AMPA and GABAa Contributions
    STN_LFP_AMPA_1 = (
        (1 / (4 * math.pi * sigma))
        * np.sum(
            (1 / (STN_recording_electrode_1_distances * 1e-6)) * STN_AMPA_i.transpose(),
            axis=0,
        )
        * 1e-6
    )
    STN_LFP_AMPA_2 = (
        (1 / (4 * math.pi * sigma))
        * np.sum(
            (1 / (STN_recording_electrode_2_distances * 1e-6)) * STN_AMPA_i.transpose(),
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

    # Simulation Label for writing model output data - uncomment to write the
    # specified variables to file
    simulation_label = "Steady_State_Simulation"
    output_dirname = os.environ.get("PYNN_OUTPUT_DIRNAME", "Simulation_Output_Results")
    simulation_output_dir = f"{output_dirname}/" + simulation_label

    if save_sim_data:
        if rank == 0:
            print("Writing data...")
        # Write population membrane voltage data to file
        Cortical_Pop.write_data(
            simulation_output_dir + "/Cortical_Pop/Cortical_Collateral_v.mat",
            "collateral(0.5).v",
            clear=False,
        )
        Cortical_Pop.write_data(
            simulation_output_dir + "/Cortical_Pop/Cortical_Soma_v.mat",
            "soma(0.5).v",
            clear=True,
        )
        Interneuron_Pop.write_data(
            simulation_output_dir + "/Interneuron_Pop/Interneuron_Soma_v.mat",
            "soma(0.5).v",
            clear=True,
        )
        STN_Pop.write_data(
            simulation_output_dir + "/STN_Pop/STN_Soma_v.mat", "soma(0.5).v", clear=True
        )
        GPe_Pop.write_data(
            simulation_output_dir + "/GPe_Pop/GPe_Soma_v.mat", "soma(0.5).v", clear=True
        )
        GPi_Pop.write_data(
            simulation_output_dir + "/GPi_Pop/GPi_Soma_v.mat", "soma(0.5).v", clear=True
        )
        Thalamic_Pop.write_data(
            simulation_output_dir + "/Thalamic_Pop/Thalamic_Soma_v.mat",
            "soma(0.5).v",
            clear=True,
        )

        # Write the STN LFP to .mat file
        STN_LFP_Block = neo.Block(name="STN_LFP")
        STN_LFP_seg = neo.Segment(name="segment_0")
        STN_LFP_Block.segments.append(STN_LFP_seg)
        STN_LFP_signal = neo.AnalogSignal(
            STN_LFP,
            units="mV",
            t_start=0 * pq.ms,
            sampling_rate=pq.Quantity(simulator.state.dt, "1/ms"),
        )
        STN_LFP_seg.analogsignals.append(STN_LFP_signal)

        w = neo.io.NeoMatlabIO(filename=simulation_output_dir + "/STN_LFP.mat")
        w.write_block(STN_LFP_Block)

        # Write LFP AMPA and GABAa conmponents to file
        STN_LFP_AMPA_Block = neo.Block(name="STN_LFP_AMPA")
        STN_LFP_AMPA_seg = neo.Segment(name="segment_0")
        STN_LFP_AMPA_Block.segments.append(STN_LFP_AMPA_seg)
        STN_LFP_AMPA_signal = neo.AnalogSignal(
            STN_LFP_AMPA,
            units="mV",
            t_start=0 * pq.ms,
            sampling_rate=pq.Quantity(simulator.state.dt, "1/ms"),
        )
        STN_LFP_AMPA_seg.analogsignals.append(STN_LFP_AMPA_signal)
        w = neo.io.NeoMatlabIO(filename=simulation_output_dir + "/STN_LFP_AMPA.mat")
        w.write_block(STN_LFP_AMPA_Block)

        STN_LFP_GABAa_Block = neo.Block(name="STN_LFP_GABAa")
        STN_LFP_GABAa_seg = neo.Segment(name="segment_0")
        STN_LFP_GABAa_Block.segments.append(STN_LFP_GABAa_seg)
        STN_LFP_GABAa_signal = neo.AnalogSignal(
            STN_LFP_GABAa,
            units="mV",
            t_start=0 * pq.ms,
            sampling_rate=pq.Quantity(simulator.state.dt, "1/ms"),
        )
        STN_LFP_GABAa_seg.analogsignals.append(STN_LFP_GABAa_signal)
        w = neo.io.NeoMatlabIO(filename=simulation_output_dir + "/STN_LFP_GABAa.mat")
        w.write_block(STN_LFP_GABAa_Block)

    if rank == 0:
        print("Steady State Simulation Done!")

    end()
