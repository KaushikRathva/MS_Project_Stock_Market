# Note these are just sample tests
# Not very comprehensive tests

from unittest import TestCase
from trytry import HMM  # Make sure the path to trytry.py is correct
import numpy as np

class TestHmm(TestCase):

    def setUp(self):
        self.n_states = 2
        self.n_symbols = 2
        self.hmm = HMM(self.n_states, self.n_symbols)
        self.hmm.transition_prob_mat = np.array([[0.6, 0.4], [0.3, 0.7]])
        self.hmm.emission_prob_mat = np.array([[0.3, 0.7], [0.4, 0.6]])
        self.hmm.stationary_dist = np.array([0.5, 0.5])
        self.observations = [0, 1, 1, 0]  # Assuming 'A' is 0 and 'B' is 1



    def test_forward(self):
        # Forward algorithm
        forw_prob = self.hmm.forward_algo(self.observations)
        forw_prob = forw_prob.sum().item()  # Get the final probability
        forw_prob = round(forw_prob, 5)
        self.assertEqual(0.05153, forw_prob)

    def test_backward(self):
        # Backward algorithmSS
        back_prob = self.hmm.backward_algo(self.observations)
        back_prob = back_prob.sum().item()  # Get the initial probability
        back_prob = round(back_prob, 5)
        self.assertEqual(0.05153, back_prob)

    # def test_train_hmm(self):
    #     # Train HMM
    #     self.hmm.train(self.observations)
        # Check if the parameters have been updated (this is a simple check)
        # self.assertFalse(np.array_equal(self.hmm.transition_prob_mat, np.array([[0.6, 0.4], [0.3, 0.7]])))
        # self.assertFalse(np.array_equal(self.hmm.emission_prob_mat, np.array([[0.3, 0.7], [0.4, 0.6]])))
        # self.assertFalse(np.array_equal(self.hmm.stationary_dist, np.array([0.5, 0.5])))

if __name__ == '__main__':
    import unittest
    unittest.main()
