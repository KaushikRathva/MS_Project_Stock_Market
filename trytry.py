import numpy as np
from itertools import product
import copy as cp
import pickle
import matplotlib.pyplot as plt

data_csv = 'nifty50_data_2020_2024.csv'

class HMM:
    def __init__(self, n_states, n_symbols):
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.transition_prob_mat = np.zeros((n_states, n_states))
        self.emission_prob_mat = np.zeros((n_states, n_symbols))
        self.stationary_dist = np.zeros(n_states)

    def train(self, data_sequence):
        self.update_phase(data_sequence)

    def predict_next_observation(self, observation_sequence):
        """
        Predict the next most probable observation symbol based on the current observation sequence
        Inputs: observation_sequence: List of observation symbols
        Outputs: next_observation: Most probable next observation symbol
        """
        A = self.transition_prob_mat
        B = self.emission_prob_mat
        lambda_hmm = self.stationary_dist

        T = len(observation_sequence)
        delta = np.zeros((T, self.n_states))

        # Initialization
        delta[0, :] = lambda_hmm * B[:, observation_sequence[0]]

        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                delta[t, j] = np.max(delta[t-1, :] * A[:, j]) * B[j, observation_sequence[t]]

        # Predict the next observation
        next_observation_prob = np.zeros(self.n_symbols)
        for j in range(self.n_states):
            next_observation_prob += delta[T-1, j] * A[j, :] @ B

        next_observation = np.argmax(next_observation_prob)

        return next_observation

    def forward_algo(self, observation_sequence, curr_state=(-1)):
        """
        Forward Algorithm for HMM
        Inputs: observation_sequence: List of observation symbols
                curr_state: Current state
        Outputs: alpha: Forward probabilities
        """
        A = self.transition_prob_mat
        B = self.emission_prob_mat
        lambda_hmm = self.stationary_dist
        if len(observation_sequence) == 1:
            return lambda_hmm[curr_state] * B[curr_state, observation_sequence[0]]
        elif curr_state == -1:
            alpha = 0
            for state in range(self.n_states):
                alpha += self.forward_algo(observation_sequence, state)
            return alpha
        else:
            alpha = 0
            for prev_state in range(self.n_states):
                alpha += self.forward_algo(observation_sequence[:-1], prev_state) * A[prev_state, curr_state] * B[curr_state, observation_sequence[-1]]
            return alpha

    def backward_algo(self, observation_sequence, curr_state=(-1)):
        """
        Backward Algorithm for HMM
        Inputs: observation_sequence: List of observation symbols
                curr_state: Current state
        Outputs: beta: Backward probabilities
        """
        A = self.transition_prob_mat
        B = self.emission_prob_mat
        if len(observation_sequence) == 1:
            return 1
        elif curr_state == -1:
            beta = 0
            for state in range(self.n_states):
                beta += self.backward_algo(observation_sequence, state)
            return beta
        else:
            beta = 0
            for next_state in range(self.n_states):
                beta += A[curr_state, next_state] * B[next_state, observation_sequence[1]] * self.backward_algo(observation_sequence[1:], next_state)
            return beta
    
    def update_phase(self, observation_sequence):
        """
        Update phase of the HMM
        Inputs: observation_sequence: List of observation symbols
        Outputs: None
        """
        alpha, beta = self.compute_alpha_beta(observation_sequence)
        gamma = self.compute_gamma(alpha, beta)
        xi = self.compute_xi(observation_sequence, alpha, beta)
        self.update_parameters(gamma, xi, observation_sequence)

    def compute_alpha_beta(self, observation_sequence):
        """
        Compute alpha and beta matrices
        Inputs: observation_sequence: List of observation symbols
        Outputs: alpha, beta: Forward and backward probabilities
        """
        alpha = np.zeros((len(observation_sequence), self.n_states))
        beta = np.zeros((len(observation_sequence), self.n_states))
        for t in range(len(observation_sequence)):
            for state in range(self.n_states):
                alpha[t, state] = self.forward_algo(observation_sequence[:t+1], state)
                beta[t, state] = self.backward_algo(observation_sequence[t:], state)
        return alpha, beta


    def compute_gamma(self, alpha, beta):
        """
        Compute gamma matrix
        Inputs: alpha, beta: Forward and backward probabilities
        Outputs: gamma: State probabilities
        """
        return alpha * beta

    def compute_xi(self, observation_sequence, alpha, beta):
        """
        Compute xi matrix
        Inputs: observation_sequence: List of observation symbols
                alpha, beta: Forward and backward probabilities
        Outputs: xi: Transition probabilities
        """
        A = self.transition_prob_mat
        B = self.emission_prob_mat
        xi = np.zeros((len(observation_sequence)-1, self.n_states, self.n_states))
        for t in range(len(observation_sequence)-1):
            for state1 in range(self.n_states):
                for state2 in range(self.n_states):
                    xi[t, state1, state2] = alpha[t, state1] * A[state1, state2] * B[state2, observation_sequence[t+1]] * beta[t+1, state2]
        return xi

    def update_parameters(self, gamma, xi, observation_sequence):
        """
        Update HMM parameters
        Inputs: gamma: State probabilities
                xi: Transition probabilities
                observation_sequence: List of observation symbols
        Outputs: None
        """
        A_new = np.zeros((self.n_states, self.n_states))
        B_new = np.zeros((self.n_states, self.n_symbols))
        lambda_new = np.zeros(self.n_states)
        for state1 in range(self.n_states):
            for state2 in range(self.n_states):
                A_new[state1, state2] = np.sum(xi[:, state1, state2]) / np.sum(gamma[:, state1])
        for state in range(self.n_states):
            for symbol in range(self.n_symbols):
                B_new[state, symbol] = np.sum(gamma[observation_sequence == symbol, state]) / np.sum(gamma[:, state])
        for state in range(self.n_states):
            lambda_new[state] = gamma[0, state]
        self.transition_prob_mat = A_new
        self.emission_prob_mat = B_new
        self.stationary_dist = lambda_new
        self.stationary_dist = lambda_new


def find_trands(data,threshhold):
    trends = []
    for i in range(len(data) - 1):  #trend 0: down, 1: same, 2: up
        dif = data[i+1]-data[i]
        if abs(dif) < threshhold:
            trends.append(1)
        elif dif < 0:
            trends.append(0)
        else:
            trends.append(2)
        # print(f"data_i+1 : {data[i+1]},data_i : {data[i]} , data_i+1 - data_i :{dif}, trend : {trends[-1]}")
    return trends

def ready_data(data_csv):
    # Load data from CSV file
    raw_data = np.genfromtxt(data_csv, delimiter=',', dtype=None, encoding=None)
    header = raw_data[0]  # First line as header
    data = raw_data[1:]  # Remaining data

    # Convert data to a dictionary with headers as keys
    data_dict = {key: [] for key in header}
    for row in data:
        for key, value in zip(header, row):
            if key == 'Date':
                data_dict[key].append(np.datetime64(value))
            else:
                data_dict[key].append(float(value))

    # # Split the data dictionary into two parts
    # split_index = len(data_dict[header[0]]) // 2
    # train_data  = {key: values[:split_index] for key, values in data_dict.items()}
    # test_data   = {key: values[split_index:] for key, values in data_dict.items()}

    # Ensure there are at least 200 rows to split
    assert len(data_dict[header[0]]) >= 200, "Not enough data to split into 100 rows each for train and test."

    # Select the first 100 rows for training and the next 100 rows for testing
    train_data = {key: values[:100] for key, values in data_dict.items()}
    test_data = {key: values[100:200] for key, values in data_dict.items()}
    
    # # Shuffle the data
    # np.random.shuffle(data)

    # Split the data into training and test sets
    # split_index = len(data) // 2
    # train_data = data[:split_index]
    # test_data = data[split_index:]

    # # Print the shapes of the datasets
    # print(f'Training data shape: {train_data.shape}')
    # print(f'Test data shape: {test_data.shape}')

    return train_data, test_data


        

def main():
    # ready_data(data_csv)
    train_data, test_data = ready_data(data_csv)
    hmm_model = HMM(5, 3)
    open_price_hmm = cp.deepcopy(hmm_model)
    open_price_train_trend = find_trands(train_data['Open'],50)
    open_price_hmm.train(open_price_train_trend)

    print("training done.")
    open_price_test_trend = find_trands(test_data['Open'],50)

    # Generate the predicted observation sequence based on the last symbol of the open_price trend
    last_observation = open_price_train_trend[-1]
    predicted_sequence = [last_observation]

    # Predict the next 10 observations
    # for _ in range(10):
    #     next_observation = open_price_hmm.predict_next_observation(predicted_sequence)
    #     predicted_sequence.append(next_observation)

    
    
    # Plot the open price test trend and predicted sequence
    plt.figure(figsize=(12, 6))
    plt.plot(open_price_test_trend, label='Actual Open Price Trend')
    plt.plot(range(len(open_price_train_trend), len(open_price_train_trend) + len(predicted_sequence)), predicted_sequence, label='Predicted Open Price Trend', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Trend')
    plt.title('Actual vs Predicted Open Price Trend')
    plt.legend()
    plt.savefig('open_price_trend_prediction.png')
    

    # Save the trained HMM model for open price
    with open('open_price_hmm.pkl', 'wb') as file:
        pickle.dump(open_price_hmm, file)

    # Save the transition probability matrix, emission probability matrix, and stationary distribution
    np.save('open_price_transition_prob_mat.npy', open_price_hmm.transition_prob_mat)
    np.save('open_price_emission_prob_mat.npy', open_price_hmm.emission_prob_mat)
    np.save('open_price_stationary_dist.npy', open_price_hmm.stationary_dist)


    # for key, value in train_data.items():
    #     print(key, value[:5])

        
    # # Train the HMM model using the training data
    # for sequence in train_data:
    #     hmm_model.train(sequence)
    
    # # Test the HMM model using the test data
    # for sequence in test_data:
    #     print(hmm_model.forward_algo(sequence))

if __name__ == '__main__':
    main()
