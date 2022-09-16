import unittest
import numpy as np
import model

class TestPopulationLoading(unittest.TestCase):
    
    def test_cortical_population_location_fixed_seed(self):
        Pop_size = 100
        steady_state_duration = 6000.0
        simulation_runtime = 32000.0
        simulation_duration = steady_state_duration + simulation_runtime + 0.01
        v_init = -68
        rng_seed = 3695
        
        network0 = model.load_network(Pop_size, steady_state_duration, simulation_duration, simulation_runtime, v_init, rng_seed)
        network1 = model.load_network(Pop_size, steady_state_duration, simulation_duration, simulation_runtime, v_init, rng_seed)
        self.assertTrue(np.all(network0[1].positions == network1[1].positions))
        
    def test_cortical_population_location_no_seed_xy_location(self):
        Pop_size = 100
        steady_state_duration = 6000.0
        simulation_runtime = 32000.0
        simulation_duration = steady_state_duration + simulation_runtime + 0.01
        v_init = -68
        rng_seed=None
        
        network0 = model.load_network(Pop_size, steady_state_duration, simulation_duration, simulation_runtime, v_init, rng_seed)
        network1 = model.load_network(Pop_size, steady_state_duration, simulation_duration, simulation_runtime, v_init, rng_seed)
        self.assertTrue(np.all(network0[1].positions[:2] == network1[1].positions[:2]))
    
    def test_cortical_population_location_no_seed_z_location_random(self):
        Pop_size = 100
        steady_state_duration = 6000.0
        simulation_runtime = 32000.0
        simulation_duration = steady_state_duration + simulation_runtime + 0.01
        v_init = -68
        rng_seed=None
        
        network0 = model.load_network(Pop_size, steady_state_duration, simulation_duration, simulation_runtime, v_init, rng_seed)
        network1 = model.load_network(Pop_size, steady_state_duration, simulation_duration, simulation_runtime, v_init, rng_seed)
        self.assertFalse(np.all(network0[1].positions[2] == network1[1].positions[2]))