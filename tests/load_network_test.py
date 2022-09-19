import unittest
import numpy as np
import pyNN.neuron.populations
import model

class TestPopulationLoading(unittest.TestCase):
    
    def setUp(self) -> None:
        self.Pop_size = 100
        self.steady_state_duration = 6000.0
        self.simulation_runtime = 32000.0
        self.simulation_duration = self.steady_state_duration + self.simulation_runtime + 0.01
        self.v_init = -68
        self.rng_seed = 3695
        self.network0 = model.load_network(self.Pop_size, self.steady_state_duration, self.simulation_duration, self.simulation_runtime, self.v_init, self.rng_seed)
        self.network1 = model.load_network(self.Pop_size, self.steady_state_duration, self.simulation_duration, self.simulation_runtime, self.v_init, self.rng_seed)
        
        self.unseeded_network0 = model.load_network(self.Pop_size, self.steady_state_duration, self.simulation_duration, self.simulation_runtime, self.v_init, None)
        self.unseeded_network1 = model.load_network(self.Pop_size, self.steady_state_duration, self.simulation_duration, self.simulation_runtime, self.v_init, None)
    
    def test_cortical_population_location_fixed_seed(self):
        self.assertTrue(np.all(self.network0[1].positions == self.network1[1].positions))
        
    def test_cortical_population_location_no_seed_xy_location(self):
        self.assertTrue(np.all(self.unseeded_network0[1].positions[:2] == self.unseeded_network1[1].positions[:2]))
    
    def test_cortical_population_location_no_seed_z_location_random(self):
        self.assertFalse(np.all(self.unseeded_network0[1].positions[2] == self.unseeded_network1[1].positions[2]))
        
    def test_population_size(self):
        for i in range(1, 8):
            self.assertTrue(isinstance(self.network0[i], pyNN.neuron.populations.Population))
            self.assertEqual(self.Pop_size, len(self.network0[i].all_cells))
            self.assertTrue(isinstance(self.network1[i], pyNN.neuron.populations.Population))
            self.assertEqual(self.Pop_size, len(self.network1[i].all_cells))
    
    def test_membrane_noise_seeded(self):
        for i in range(0, self.Pop_size):
            self.assertTrue(np.all(self.network0[-1][i].amplitudes == self.network1[-1][i].amplitudes))
            self.assertTrue(np.all(self.network0[-2][i].amplitudes == self.network1[-2][i].amplitudes))
    
    def test_membrane_noise_unseeded(self):
        for i in range(0, self.Pop_size):
            self.assertFalse(np.all(self.unseeded_network0[-1][i].amplitudes == self.unseeded_network1[-1][i].amplitudes))
            self.assertFalse(np.all(self.unseeded_network0[-2][i].amplitudes == self.unseeded_network1[-2][i].amplitudes))