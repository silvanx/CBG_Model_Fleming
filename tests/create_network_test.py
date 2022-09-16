import unittest
import numpy as np
import model
import pyNN.neuron.populations

class TestPopulationCreating(unittest.TestCase):
    
    
    def setUp(self) -> None:
        self.Pop_size = 100
        steady_state_duration = 6000.0
        simulation_runtime = 32000.0
        simulation_duration = steady_state_duration + simulation_runtime + 0.01
        v_init = -68
        rng_seed = 3695
        
        self.network0 = model.create_network(self.Pop_size, steady_state_duration, simulation_duration, simulation_runtime, v_init, rng_seed)
        self.network1 = model.create_network(self.Pop_size, steady_state_duration, simulation_duration, simulation_runtime, v_init, rng_seed)
    
    
    def test_cortical_population_location_fixed_seed(self):
        self.assertTrue(np.all(self.network0[1].positions == self.network1[1].positions))
        
    def test_cortical_population_location_no_seed_xyz_location_random(self):
        steady_state_duration = 6000.0
        simulation_runtime = 32000.0
        simulation_duration = steady_state_duration + simulation_runtime + 0.01
        v_init = -68
        rng_seed = None
        
        self.network0 = model.create_network(self.Pop_size, steady_state_duration, simulation_duration, simulation_runtime, v_init, rng_seed)
        self.network1 = model.create_network(self.Pop_size, steady_state_duration, simulation_duration, simulation_runtime, v_init, rng_seed)
        
        self.assertFalse(np.all(self.network0[1].positions[0] == self.network1[1].positions[0]))
        self.assertFalse(np.all(self.network0[1].positions[1] == self.network1[1].positions[1]))
        self.assertFalse(np.all(self.network0[1].positions[2] == self.network1[1].positions[2]))
        
    def test_population_size(self):
        for i in range(1, 8):
            self.assertTrue(isinstance(self.network0[i], pyNN.neuron.populations.Population))
            self.assertEqual(self.Pop_size, len(self.network0[i].all_cells))
            self.assertTrue(isinstance(self.network1[i], pyNN.neuron.populations.Population))
            self.assertEqual(self.Pop_size, len(self.network1[i].all_cells))