# Note these are just sample tests
# Not very comprehensive tests

from unittest import TestCase
from trytry_para_gpu import HMM
import numpy as np

# states = ['Rainy', 'Sunny']
# possible_observation = ['Walk', 'Shop']

states = ('s', 't')
possible_observation = ('A','B' )

# The observations that we observe and feed to the model
# observations = [0, 1, 1, 0]  # Assuming 'Walk' is 0 and 'Shop' is 1
observations = [0, 1, 1, 0]  # Assuming 'A' is 0 and 'B' is 1
obs4 = [1, 0, 1]

# Tuple of observations
observation_tuple = []
observation_tuple.extend([observations, obs4])
print(observation_tuple)
quantities_observations = [10, 20]
start_probability = np.array([0.5, 0.5])
transition_probability = np.array([[0.6, 0.4], [0.3, 0.7]])
emission_probability = np.array([[0.3, 0.7], [0.4, 0.6]])

class TestHmm(TestCase):

    def test_forward(self):
        # Declare Class object
        test = HMM(states, possible_observation)
        test.transition_prob_mat = np.array([[0.6, 0.4], [0.3, 0.7]])
        test.emission_prob_mat = np.array([[0.3, 0.7], [0.4, 0.6]])
        test.stationary_dist = np.array([0.5, 0.5])

        # # Forward algorithm
        # forw_prob = test.forward_algo(observations)
        # forw_prob = forw_prob[-1].sum().item()  # Get the final probability
        # forw_prob = round(forw_prob, 5)
        # self.assertEqual(0.05153, forw_prob)

        print(observations)
        forw_prob = test.forward_algo(observations)
        print(forw_prob)
        forw_prob = forw_prob[-1].sum().item()  # Convert the final probability from torch tensor to a scalar
        forw_prob = round(forw_prob, 5)
        self.assertEqual(0.05153, forw_prob)

    def test_backward(self):
        # Declare Class object
        test = HMM(states, possible_observation)
        test.transition_prob_mat = np.array([[0.6, 0.4], [0.3, 0.7]])
        test.emission_prob_mat = np.array([[0.3, 0.7], [0.4, 0.6]])
        test.stationary_dist = np.array([0.5, 0.5])

        # Backward algorithm
        print(observations)
        back_prob = test.backward_algo(observations)
        print(back_prob)
        back_prob = back_prob[0].sum().item()  # Get the initial probability
        back_prob = round(back_prob, 5)
        expected_back_prob = 0.169272  # Replace this with the correct expected value if different
        self.assertEqual(expected_back_prob, back_prob)


    # def test_train_hmm(self):
    #     # Declare Class object
    #     test = HMM(states, possible_observation)
    #     test.transition_prob_mat = np.array([[0.6, 0.4], [0.3, 0.7]])
    #     test.emission_prob_mat = np.array([[0.3, 0.7], [0.4, 0.6]])
    #     test.stationary_dist = np.array([0.5, 0.5])

    #     # Train HMM
    #     test.train(observations)
    #     # Check if the parameters have been updated (this is a simple check)
    #     self.assertFalse(np.array_equal(test.transition_prob_mat, np.array([[0.6, 0.4], [0.3, 0.7]])))
    #     self.assertFalse(np.array_equal(test.emission_prob_mat, np.array([[0.3, 0.7], [0.4, 0.6]])))
    #     self.assertFalse(np.array_equal(test.stationary_dist, np.array([0.5, 0.5])))

if __name__ == '__main__':
    import unittest
    unittest.main()