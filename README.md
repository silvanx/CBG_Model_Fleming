# About
This repository contains a computational model of the cortico-basal ganglia loop, capable of recording local field
potentials (LFP) from the subthalamic nucleus (STN), as well as delivering deep brain stimulation (DBS) in the same
location. The original model was presented in the article by Fleming et al.,
available at https://doi.org/10.3171/2023.2.JNS222576

# Changes from the original model
- The model run twice with the same random seed produces numerically identical results
- This model is MPI-enabled, which means it can be run in parallel, reducing the simulation time
- The simulation parameters are set with a config file that is passed as a command line argument
  to the `run_model.py` script
- Ability to record cortical "LFP" analog
- Possibility to choose if the cortical beta comes intracellularly or synaptically

# Running the model using Docker
First, install Docker (https://www.docker.com/products/docker-desktop/). Then clone the project
```
git clone https://github.com/silvanx/CBG_Model_Fleming.git 
cd CBG_Model_Fleming
```
Build the image using included dockerfile
```
docker build -t cbg-model .
```
Run the code in the container
```
docker run --name container-name cbg-model <config_file>.yml
```
If you don't specify the config file (relative to the Cortex_BasalGanglia_DBS_model directory),
by default ```conf_zero_4s.yml``` will be run.

Get the simulation data out from the container
```
docker cp container-name:/usr/app/src/CBG_Fleming_Model/RESULTS ./simulation-results
```

Several scripts to help with data loading and plotting are available in the following repository:
https://github.com/silvanx/FlemingModelResultAnalysis

If you want to run the simulations directly from your OS, use the dockerfile as the guide with respect to dependencies
and their required versions.

# Config file
The config file specifies the parameters of the simulation 
## Simulation 
- `RandomSeed`: seed for the RNG
- `TimeStep`: timestep of the NEURON simulator
- `SteadyStateDuration`: how long to wait before applying stimulation; unit: ms
- `RunTime`: how long to run the simulation *after* steady state; unit: ms
- `save_stn_voltage`: whether to write STN neuron membrane voltage to a file; True/False
- `save_gpe_voltage`: whether to write GPe neuron membrane voltage to a file; True/False
- `save_ctx_voltage`: whether to write cortical neuron membrane voltage to a file; True/False
- `save_ctx_lfp`: whether to write cortical neuron synaptic currents to a file; True/False
## Model
- `Pop_size`: how many neurons per cell population
- `create_new_network`: should I create a new model or read the structure from a file?; True/False (default: False = read structure from a file)
- `ctx_dc_offset`: constant current applied to cortical neurons; unit: nA
- `ctx_dc_offset_std`: add gaussian random variability to the constant cortical current by specifying the standard deviation of the distribution; unit: nA

- `cortical_beta_mechanism`: how to introduce the beta activity into the cortex; allowed values:
  - `current`: intracellular current modulating the firing rate of the Cortical neurons
  - `spikes`: synaptic input with beta frequency

If `cortical beta mechanism` is set to `spikes`, the following parameters are responsible for the behaviour of the system:
- `ctx_beta_spike_frequency`: Frequency of the Poisson-distributed spikes coming into the cortex
- `ctx_beta_synapse_strength`: Strength of the synaptic connections
- `ctx_beta_spike_isi_dither`: Standard deviation of additional jitter added to spike times

If `cortical beta mechanism` is set to `current`, the following parameters are responsible for the behaviour of the system:
- `ctx_slow_modulation_step_count`: how many times during the simulation to switch ctx input from `0` to `ctx_slow_modulation_amplitude`
- `ctx_slow_modulation_amplitude`: amplitude of slow cortical modulation
- `beta_burst_modulation_scale`: amplitude of the modulating current in cortical neurons; unit: nA
## Controller
- `Controller`: specify the controller type; allowed values:
  - `ZERO`: no stimulation
  - `OPEN`: constant open-loop stimulation
  - `PID`: PID (proportional-integral-derivative) controller
  - `IFT`: Iterative Feedback Tuning controller (EXPERIMENTAL)
- `setpoint`: target value of the biomarker
- `ts`: time between subsequent controller calls; unit: s
- `min_value`: minimum value of controller output
- `max_value`: maximum value of controller output
- `controller_window_length`: length of time on which the biomarker value is calculated; unit: ms
- `controller_window_tail_length`: length of the tail which is ignored for calculating biomarker to avoid edge effects; unit: ms
- `stimulation_frequency`: stimulation frequency; unit: Hz; default: 130.0
### PID controller settings
- `kp`: proportional gain
- `ti`: integral constant
- `td`: derivative constant

### IFT controller settings
- `kp`: initial proportional gain
- `ti`: initial integral constant
- `stage_length`: duration of the "experiment"; unit: s
- `gamma`: regulates gradient descent speed
- `lam`: lambda parameter, weighting the MSE and TEED parts of the fitness function
- `min_kp`: minimal value of kp
- `min_ti`: minimal value of ti 
- `fix_kp`: if True, don't update the value of Kp (default: False)
- `fix_ti`: if True, don't update the value of Ti (default: False)

### OPEN controller settings
- `stimulation_amplitude`: amplitude of the open-loop stimulation
