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

import neuron
from pyNN.neuron import setup, run_to_steady_state, end
from pyNN.parameters import Sequence
import numpy as np
import math

# Import global variables for GPe DBS
import Global_Variables as GV
from utils import make_beta_cheby1_filter
from model import load_network, electrode_distance

h = neuron.h


if __name__ == '__main__':
    rng_seed = 3695
    timestep = 0.01
    # Setup simulation
    setup(timestep=timestep, rngseed=rng_seed)
    steady_state_duration = 6000.0  # Duration of simulation steady state
    simulation_duration = steady_state_duration  # Total simulation time
    rec_sampling_interval = 0.5  # Fs = 2000Hz
    Pop_size = 100

    # Make beta band filter centred on 25Hz (cutoff frequencies are 21-29 Hz)
    # for biomarker estimation
    fs = (1000 / rec_sampling_interval)
    beta_b, beta_a = make_beta_cheby1_filter(fs=fs, n=4, rp=0.5,
                                             low=21, high=29)

    # Use CVode to calculate i_membrane_ for fast LFP calculation
    cvode = h.CVode()
    cvode.active(0)

    # Second spatial derivative (the segment current) for the collateral
    cvode.use_fast_imem(1)

    # Set initial values for cell membrane voltages
    v_init = -68

    (striatal_spike_times,
     Cortical_Pop, Interneuron_Pop, STN_Pop, GPe_Pop, GPi_Pop,
     Striatal_Pop, Thalamic_Pop,
     prj_CorticalAxon_Interneuron, prj_Interneuron_CorticalSoma,
     prj_CorticalSTN, prj_STNGPe, prj_GPeGPe, prj_GPeSTN,
     prj_StriatalGPe, prj_STNGPi, prj_GPeGPi, prj_GPiThalamic,
     prj_ThalamicCortical, prj_CorticalThalamic, GPe_stimulation_order,
     _, _) = load_network(Pop_size, steady_state_duration, simulation_duration,
                          simulation_duration, v_init)

    # # Create random distribution for cell membrane noise current
    # r_init = RandomDistribution('uniform', (0, Pop_size))

    # # Create Spaces for STN Population
    # STN_Electrode_space = space.Space(axes='xy')
    # STN_space = space.RandomStructure(boundary=space.Sphere(2000))  # Sphere with radius 2000um

    # Generate poisson distributed spike time striatal input
    # striatal_spike_times = generate_poisson_spike_times(Pop_size, steady_state_duration, simulation_duration, 20, 1.0,
    #                                                     3695)

    # Save/load Striatal Spike times
    # np.save('Striatal_Spike_Times.npy', striatal_spike_times)  # Save spike times so they can be reloaded
    # striatal_spike_times = np.load('Striatal_Spike_Times.npy', allow_pickle=True)  # Load spike times from file

    # Generate the cortico-basal ganglia neuron populations
    # Cortical_Pop = Population(Pop_size, Cortical_Neuron_Type(soma_bias_current_amp=0.245), structure=STN_space,
    #                           label='Cortical Neurons')  # Better than above (ibias=0.2575)
    # Interneuron_Pop = Population(Pop_size, Interneuron_Type(bias_current_amp=0.070), initial_values={'v': v_init},
    #                              label='Interneurons')
    # STN_Pop = Population(Pop_size, STN_Neuron_Type(bias_current=-0.125), structure=STN_space,
    #                      initial_values={'v': v_init}, label='STN Neurons')
    # GPe_Pop = Population(Pop_size, GP_Neuron_Type(bias_current=-0.009), initial_values={'v': v_init},
    #                      label='GPe Neurons')  # GPe/i have the same parameters, but different bias currents
    # GPi_Pop = Population(Pop_size, GP_Neuron_Type(bias_current=0.006), initial_values={'v': v_init},
    #                      label='GPi Neurons')  # GPe/i have the same parameters, but different bias currents
    # Striatal_Pop = Population(Pop_size, SpikeSourceArray(spike_times=striatal_spike_times[0][0]),
    #                           label='Striatal Neuron Spike Source')
    # Thalamic_Pop = Population(Pop_size, Thalamic_Neuron_Type(), initial_values={'v': v_init}, label='Thalamic Neurons')

    # Load Cortical Bias currents for beta burst modulation - turn bursts off when running model to steady state
    # burst_times_script = "burst_times_1.txt"
    # burst_level_script = "burst_level_1.txt"
    # modulation_times = np.loadtxt(burst_times_script, delimiter=',')
    # modulation_signal = np.loadtxt(burst_level_script, delimiter=',')
    # modulation_signal = 0.00 * modulation_signal  # Scale the modulation signal - turn off
    # cortical_modulation_current = StepCurrentSource(times=modulation_times, amplitudes=modulation_signal)
    # Cortical_Pop.inject(cortical_modulation_current)

    # Generate Noisy current sources for cortical pyramidal and interneuron populations
    # Cortical_Pop_Membrane_Noise = [NoisyCurrentSource(mean=0, stdev=0.005, start=0.0, stop=simulation_duration, dt=1.0)
    #                                for count in range(Pop_size)]
    # Interneuron_Pop_Membrane_Noise = [NoisyCurrentSource(mean=0, stdev=0.005, start=0.0, stop=simulation_duration,
    #                                                      dt=1.0)
    #                                   for count in range(Pop_size)]

    # Inject each membrane noise current into each cortical and interneuron in network
    # for Cortical_Neuron, Cortical_Neuron_Membrane_Noise in zip(Cortical_Pop, Cortical_Pop_Membrane_Noise):
    #     Cortical_Neuron.inject(Cortical_Neuron_Membrane_Noise)

    # for Interneuron, Interneuron_Membrane_Noise in zip(Interneuron_Pop, Interneuron_Pop_Membrane_Noise):
    #     Interneuron.inject(Interneuron_Membrane_Noise)

    # Update the spike times for the striatal populations
    # for i in range(0, Pop_size):
    #     Striatal_Pop[i].spike_times = striatal_spike_times[i][0]

    # # Load cortical positions - Comment/Remove to generate new positions
    # Cortical_Neuron_xy_Positions = np.loadtxt('cortical_xy_pos.txt', delimiter=',')
    # Cortical_Neuron_x_Positions = Cortical_Neuron_xy_Positions[0, :]
    # Cortical_Neuron_y_Positions = Cortical_Neuron_xy_Positions[1, :]

    # # Set cortical xy positions to those loaded in
    # for cell_id, Cortical_cell in enumerate(Cortical_Pop):
    #     Cortical_cell.position[0] = Cortical_Neuron_x_Positions[cell_id]
    #     Cortical_cell.position[1] = Cortical_Neuron_y_Positions[cell_id]

    # # Load STN positions - Comment/Remove to generate new positions
    # STN_Neuron_xy_Positions = np.loadtxt('STN_xy_pos.txt', delimiter=',')
    # STN_Neuron_x_Positions = STN_Neuron_xy_Positions[0, :]
    # STN_Neuron_y_Positions = STN_Neuron_xy_Positions[1, :]

    # # Set STN xy positions to those loaded in
    # for cell_id, STN_cell in enumerate(STN_Pop):
    #     STN_cell.position[0] = STN_Neuron_x_Positions[cell_id]
    #     STN_cell.position[1] = STN_Neuron_y_Positions[cell_id]
    #     STN_cell.position[2] = 500

    # Assign Positions for recording and stimulating electrode point sources
    # recording_electrode_1_position = np.array([0, -1500, 250])
    # recording_electrode_2_position = np.array([0, 1500, 250])
    # stimulating_electrode_position = np.array([0, 0, 250])

    # Calculate STN cell distances to each recording electrode - only using xy coordinates for distance calculations
    # STN_recording_electrode_1_distances = distances_to_electrode(recording_electrode_1_position, STN_Pop)
    # STN_recording_electrode_2_distances = distances_to_electrode(recording_electrode_2_position, STN_Pop)

    # Calculate Cortical Collateral distances from the stimulating electrode - uses xyz coordinates for distance calculation - these distances need to be in um for xtra
    # Cortical_Collateral_stimulating_electrode_distances = collateral_distances_to_electrode(
    #     stimulating_electrode_position, Cortical_Pop, L=500, nseg=11)
    # np.savetxt('cortical_collateral_electrode_distances.txt', Cortical_Collateral_stimulating_electrode_distances, delimiter=',')	# Save the generated cortical collateral stimulation electrode distances to a textfile

    # Synaptic Connections
    # Add variability to Cortical connections - cortical interneuron connection weights are random from uniform distribution
    # gCtxInt_max_weight = 2.5e-3  # Ctx -> Int max coupling value
    # gIntCtx_max_weight = 6.0e-3  # Int -> Ctx max coupling value
    # gCtxInt = RandomDistribution('uniform', (0, gCtxInt_max_weight), rng=NumpyRNG(seed=3695))
    # gIntCtx = RandomDistribution('uniform', (0, gIntCtx_max_weight), rng=NumpyRNG(seed=3695))

    # # Define other synaptic connection weights and delays
    # syn_CorticalAxon_Interneuron = StaticSynapse(weight=gCtxInt, delay=2)
    # syn_Interneuron_CorticalSoma = StaticSynapse(weight=gIntCtx, delay=2)
    # syn_CorticalSpikeSourceCorticalAxon = StaticSynapse(weight=0.25, delay=0)
    # syn_CorticalCollateralSTN = StaticSynapse(weight=0.12, delay=1)
    # syn_STNGPe = StaticSynapse(weight=0.111111, delay=4)
    # syn_GPeGPe = StaticSynapse(weight=0.015, delay=4)
    # syn_GPeSTN = StaticSynapse(weight=0.111111, delay=3)
    # syn_StriatalGPe = StaticSynapse(weight=0.01, delay=1)
    # syn_STNGPi = StaticSynapse(weight=0.111111, delay=2)
    # syn_GPeGPi = StaticSynapse(weight=0.111111, delay=2)
    # syn_GPiThalamic = StaticSynapse(weight=3.0, delay=2)
    # syn_ThalamicCortical = StaticSynapse(weight=5, delay=2)
    # syn_CorticalThalamic = StaticSynapse(weight=0.0, delay=2)

    # # Load network topology from file
    # prj_CorticalAxon_Interneuron = Projection(Cortical_Pop, Interneuron_Pop,
    #                                           FromFileConnector("CorticalAxonInterneuron_Connections.txt"),
    #                                           syn_CorticalAxon_Interneuron, source='middle_axon_node',
    #                                           receptor_type='AMPA')
    # prj_Interneuron_CorticalSoma = Projection(Interneuron_Pop, Cortical_Pop,
    #                                           FromFileConnector("InterneuronCortical_Connections.txt"),
    #                                           syn_Interneuron_CorticalSoma, receptor_type='GABAa')
    # prj_CorticalSTN = Projection(Cortical_Pop, STN_Pop, FromFileConnector("CorticalSTN_Connections.txt"),
    #                              syn_CorticalCollateralSTN, source='collateral(0.5)', receptor_type='AMPA')
    # prj_STNGPe = Projection(STN_Pop, GPe_Pop, FromFileConnector("STNGPe_Connections.txt"), syn_STNGPe,
    #                         source='soma(0.5)', receptor_type='AMPA')
    # prj_GPeGPe = Projection(GPe_Pop, GPe_Pop, FromFileConnector("GPeGPe_Connections.txt"), syn_GPeGPe,
    #                         source='soma(0.5)', receptor_type='GABAa')
    # prj_GPeSTN = Projection(GPe_Pop, STN_Pop, FromFileConnector("GPeSTN_Connections.txt"), syn_GPeSTN,
    #                         source='soma(0.5)', receptor_type='GABAa')
    # prj_StriatalGPe = Projection(Striatal_Pop, GPe_Pop, FromFileConnector("StriatalGPe_Connections.txt"),
    #                              syn_StriatalGPe, source='soma(0.5)', receptor_type='GABAa')
    # prj_STNGPi = Projection(STN_Pop, GPi_Pop, FromFileConnector("STNGPi_Connections.txt"), syn_STNGPi,
    #                         source='soma(0.5)', receptor_type='AMPA')
    # prj_GPeGPi = Projection(GPe_Pop, GPi_Pop, FromFileConnector("GPeGPi_Connections.txt"), syn_GPeGPi,
    #                         source='soma(0.5)', receptor_type='GABAa')
    # prj_GPiThalamic = Projection(GPi_Pop, Thalamic_Pop, FromFileConnector("GPiThalamic_Connections.txt"),
    #                              syn_GPiThalamic, source='soma(0.5)', receptor_type='GABAa')
    # prj_ThalamicCortical = Projection(Thalamic_Pop, Cortical_Pop,
    #                                   FromFileConnector("ThalamicCorticalSoma_Connections.txt"), syn_ThalamicCortical,
    #                                   source='soma(0.5)', receptor_type='AMPA')
    # prj_CorticalThalamic = Projection(Cortical_Pop, Thalamic_Pop,
    #                                   FromFileConnector("CorticalSomaThalamic_Connections.txt"), syn_CorticalThalamic,
    #                                   source='soma(0.5)', receptor_type='AMPA')

    # Define state variables to record from each population
    Cortical_Pop.record('soma(0.5).v', sampling_interval=rec_sampling_interval)
    Cortical_Pop.record('collateral(0.5).v', sampling_interval=rec_sampling_interval)
    Interneuron_Pop.record('soma(0.5).v', sampling_interval=rec_sampling_interval)
    STN_Pop.record('soma(0.5).v', sampling_interval=rec_sampling_interval)
    STN_Pop.record('AMPA.i', sampling_interval=rec_sampling_interval)
    STN_Pop.record('GABAa.i', sampling_interval=rec_sampling_interval)
    Striatal_Pop.record('spikes')
    GPe_Pop.record('soma(0.5).v', sampling_interval=rec_sampling_interval)
    GPi_Pop.record('soma(0.5).v', sampling_interval=rec_sampling_interval)
    Thalamic_Pop.record('soma(0.5).v', sampling_interval=rec_sampling_interval)

    # Assign Positions for recording and stimulating electrode point sources
    recording_electrode_1_position = np.array([0, -1500, 250])
    recording_electrode_2_position = np.array([0, 1500, 250])
    stimulating_electrode_position = np.array([0, 0, 250])

    (STN_recording_electrode_1_distances,
     STN_recording_electrode_2_distances,
     Cortical_Collateral_stimulating_electrode_distances
     ) = electrode_distance(recording_electrode_1_position,
                            recording_electrode_2_position, STN_Pop,
                            stimulating_electrode_position, Cortical_Pop)

    # Conductivity and resistivity values for homogenous, isotropic medium
    sigma = 0.27  # Latikka et al. 2001 - Conductivity of Brain tissue S/m
    # rho needs units of ohm cm for xtra mechanism (S/m -> S/cm)
    rho = (1 / (sigma * 1e-2))

    # Calculate transfer resistances for each collateral segment for xtra
    # units are Mohms
    collateral_rx = (0.01 * (rho / (4 * math.pi)) *
                     (1 / Cortical_Collateral_stimulating_electrode_distances))

    # Convert ndarray to array of Sequence objects - needed to set cortical
    # collateral transfer resistances
    collateral_rx_seq = np.ndarray(shape=(1, Pop_size),
                                   dtype=Sequence).flatten()
    for ii in range(0, Pop_size):
        collateral_rx_seq[ii] = Sequence(collateral_rx[ii, :].flatten())

    # Assign transfer resistances values to collaterals
    for ii, cortical_neuron in enumerate(Cortical_Pop):
        cortical_neuron.collateral_rx = collateral_rx_seq[ii]

    # Initialise STN LFP list
    STN_LFP = []
    STN_LFP_AMPA = []
    STN_LFP_GABAa = []

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
    for i in range(0, Pop_size):
        # Neuron vector of GPe DBS signals
        GPe_DBS_Signal_neuron.append(h.Vector([0, 0]))
        GPe_DBS_times_neuron.append(h.Vector([0, steady_state_duration + 10]))

        # Play the stimulation into eacb GPe neuron
        GPe_DBS_Signal_neuron[i].play(GV.GPe_stimulation_iclamps[i]._ref_amp,
                                      GPe_DBS_times_neuron[i], 1)

    # Run the model to the steady state
    run_to_steady_state(steady_state_duration)

    # Calculate the LFP and biomarkers, etc.
    STN_AMPA_i = np.array(
        STN_Pop.get_data('AMPA.i').segments[0].analogsignals[0])
    STN_GABAa_i = np.array(
        STN_Pop.get_data('GABAa.i').segments[0].analogsignals[0])
    STN_Syn_i = STN_AMPA_i + STN_GABAa_i

    # STN LFP Calculation - Syn_i is in units of nA -> LFP units are mV
    STN_LFP_1 = (1 / (4 * math.pi * sigma)) * np.sum(
        (1 / (STN_recording_electrode_1_distances * 1e-6)) * STN_Syn_i.transpose(), axis=0) * 1e-6
    STN_LFP_2 = (1 / (4 * math.pi * sigma)) * np.sum(
        (1 / (STN_recording_electrode_2_distances * 1e-6)) * STN_Syn_i.transpose(), axis=0) * 1e-6
    STN_LFP = STN_LFP_1 - STN_LFP_2

    # STN LFP AMPA and GABAa Contributions
    STN_LFP_AMPA_1 = (1 / (4 * math.pi * sigma)) * np.sum(
        (1 / (STN_recording_electrode_1_distances * 1e-6)) * STN_AMPA_i.transpose(), axis=0) * 1e-6
    STN_LFP_AMPA_2 = (1 / (4 * math.pi * sigma)) * np.sum(
        (1 / (STN_recording_electrode_2_distances * 1e-6)) * STN_AMPA_i.transpose(), axis=0) * 1e-6
    STN_LFP_AMPA = STN_LFP_AMPA_1 - STN_LFP_AMPA_2
    STN_LFP_GABAa_1 = (1 / (4 * math.pi * sigma)) * np.sum(
        (1 / (STN_recording_electrode_1_distances * 1e-6)) * STN_GABAa_i.transpose(), axis=0) * 1e-6
    STN_LFP_GABAa_2 = (1 / (4 * math.pi * sigma)) * np.sum(
        (1 / (STN_recording_electrode_2_distances * 1e-6)) * STN_GABAa_i.transpose(), axis=0) * 1e-6
    STN_LFP_GABAa = STN_LFP_GABAa_1 - STN_LFP_GABAa_2

    # Simulation Label for writing model output data - uncomment to write the specified variables to file
    simulation_label = "Steady_State_Simulation"
    simulation_output_dir = "Simulation_Output_Results/" + simulation_label

    # # Write population membrane voltage data to file
    # Cortical_Pop.write_data(simulation_output_dir + "/Cortical_Pop/Cortical_Collateral_v.mat", 'collateral(0.5).v', clear=False)
    # Cortical_Pop.write_data(simulation_output_dir + "/Cortical_Pop/Cortical_Soma_v.mat", 'soma(0.5).v', clear=True)
    # Interneuron_Pop.write_data(simulation_output_dir + "/Interneuron_Pop/Interneuron_Soma_v.mat", 'soma(0.5).v', clear=True)
    # STN_Pop.write_data(simulation_output_dir + "/STN_Pop/STN_Soma_v.mat", 'soma(0.5).v', clear=True)
    # GPe_Pop.write_data(simulation_output_dir + "/GPe_Pop/GPe_Soma_v.mat", 'soma(0.5).v', clear=True)
    # GPi_Pop.write_data(simulation_output_dir + "/GPi_Pop/GPi_Soma_v.mat", 'soma(0.5).v', clear=True)
    # Thalamic_Pop.write_data(simulation_output_dir + "/Thalamic_Pop/Thalamic_Soma_v.mat", 'soma(0.5).v', clear=True)
    #
    # # Write the STN LFP to .mat file
    # STN_LFP_Block = neo.Block(name='STN_LFP')
    # STN_LFP_seg = neo.Segment(name='segment_0')
    # STN_LFP_Block.segments.append(STN_LFP_seg)
    # STN_LFP_signal = neo.AnalogSignal(STN_LFP, units='mV', t_start=0*pq.ms, sampling_rate=pq.Quantity(simulator.state.dt, '1/ms'))
    # STN_LFP_seg.analogsignals.append(STN_LFP_signal)
    #
    # w = neo.io.NeoMatlabIO(filename=simulation_output_dir + "/STN_LFP.mat")
    # w.write_block(STN_LFP_Block)
    #
    # # Write LFP AMPA and GABAa conmponents to file
    # STN_LFP_AMPA_Block = neo.Block(name='STN_LFP_AMPA')
    # STN_LFP_AMPA_seg = neo.Segment(name='segment_0')
    # STN_LFP_AMPA_Block.segments.append(STN_LFP_AMPA_seg)
    # STN_LFP_AMPA_signal = neo.AnalogSignal(STN_LFP_AMPA, units='mV', t_start=0*pq.ms, sampling_rate=pq.Quantity(simulator.state.dt, '1/ms'))
    # STN_LFP_AMPA_seg.analogsignals.append(STN_LFP_AMPA_signal)
    # w = neo.io.NeoMatlabIO(filename=simulation_output_dir + "/STN_LFP_AMPA.mat")
    # w.write_block(STN_LFP_AMPA_Block)
    #
    # STN_LFP_GABAa_Block = neo.Block(name='STN_LFP_GABAa')
    # STN_LFP_GABAa_seg = neo.Segment(name='segment_0')
    # STN_LFP_GABAa_Block.segments.append(STN_LFP_GABAa_seg)
    # STN_LFP_GABAa_signal = neo.AnalogSignal(STN_LFP_GABAa, units='mV', t_start=0*pq.ms, sampling_rate=pq.Quantity(simulator.state.dt, '1/ms'))
    # STN_LFP_GABAa_seg.analogsignals.append(STN_LFP_GABAa_signal)
    # w = neo.io.NeoMatlabIO(filename=simulation_output_dir + "/STN_LFP_GABAa.mat")
    # w.write_block(STN_LFP_GABAa_Block)

    print("Steady State Simulation Done!")

    end()
