import numpy as np
from itertools import product
import copy as cp
import pickle
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

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
        A = self.transition_prob_mat
        B = self.emission_prob_mat
        lambda_hmm = self.stationary_dist

        T = len(observation_sequence)
        delta = np.zeros((T, self.n_states))

        delta[0, :] = lambda_hmm * B[:, observation_sequence[0]]

        for t in range(1, T):
            for j in range(self.n_states):
                delta[t, j] = np.max(delta[t-1, :] * A[:, j]) * B[j, observation_sequence[t]]

        next_observation_prob = np.zeros(self.n_symbols)
        for j in range(self.n_states):
            next_observation_prob += delta[T-1, j] * A[j, :] @ B

        next_observation = np.argmax(next_observation_prob)

        return next_observation

    def forward_algo(self, observation_sequence, curr_state=(-1)):
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
        alpha, beta = self.compute_alpha_beta(observation_sequence)
        gamma = self.compute_gamma(alpha, beta)
        xi = self.compute_xi(observation_sequence, alpha, beta)
        self.update_parameters(gamma, xi, observation_sequence)

    def compute_alpha_beta(self, observation_sequence):
        alpha = np.zeros((len(observation_sequence), self.n_states))
        beta = np.zeros((len(observation_sequence), self.n_states))
        for t in range(len(observation_sequence)):
            for state in range(self.n_states):
                alpha[t, state] = self.forward_algo(observation_sequence[:t+1], state)
                beta[t, state] = self.backward_algo(observation_sequence[t:], state)
        return alpha, beta

    def compute_gamma(self, alpha, beta):
        return alpha * beta

    def compute_xi(self, observation_sequence, alpha, beta):
        A = self.transition_prob_mat
        B = self.emission_prob_mat
        xi = np.zeros((len(observation_sequence)-1, self.n_states, self.n_states))
        for t in range(len(observation_sequence)-1):
            for state1 in range(self.n_states):
                for state2 in range(self.n_states):
                    xi[t, state1, state2] = alpha[t, state1] * A[state1, state2] * B[state2, observation_sequence[t+1]] * beta[t+1, state2]
        return xi

    def update_parameters(self, gamma, xi, observation_sequence):
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

def find_trands(data,threshhold):
    trends = []
    for i in range(len(data) - 1):
        dif = data[i+1]-data[i]
        if abs(dif) < threshhold:
            trends.append(1)
        elif dif < 0:
            trends.append(0)
        else:
            trends.append(2)
    return trends

def ready_data(data_csv):
    raw_data = np.genfromtxt(data_csv, delimiter=',', dtype=None, encoding=None)
    header = raw_data[0]
    data = raw_data[1:]

    data_dict = {key: [] for key in header}
    for row in data:
        for key, value in zip(header, row):
            if key == 'Date':
                data_dict[key].append(np.datetime64(value))
            else:
                data_dict[key].append(float(value))

    assert len(data_dict[header[0]]) >= 200, "Not enough data to split into 100 rows each for train and test."

    train_data = {key: values[:100] for key, values in data_dict.items()}
    test_data = {key: values[100:200] for key, values in data_dict.items()}

    return train_data, test_data

def train_hmm(hmm_model, train_data, key, threshold):
    train_trend = find_trands(train_data[key], threshold)
    hmm_model.train(train_trend)
    return hmm_model, train_trend

def predict_hmm(hmm_model, train_trend, num_predictions=10):
    last_observation = train_trend[-1]
    predicted_sequence = [last_observation]

    for _ in range(num_predictions):
        next_observation = hmm_model.predict_next_observation(predicted_sequence)
        predicted_sequence.append(next_observation)

    return predicted_sequence

def main():
    train_data, test_data = ready_data(data_csv)
    hmm_model = HMM(5, 3)
    open_price_hmm = cp.deepcopy(hmm_model)

    with ThreadPoolExecutor() as executor:
        future_train = executor.submit(train_hmm, open_price_hmm, train_data, 'Open', 50)
        open_price_hmm, open_price_train_trend = future_train.result()

        open_price_test_trend = find_trands(test_data['Open'], 50)

        future_predict = executor.submit(predict_hmm, open_price_hmm, open_price_train_trend, 10)
        predicted_sequence = future_predict.result()

    plt.figure(figsize=(12, 6))
    plt.plot(open_price_test_trend, label='Actual Open Price Trend')
    plt.plot(range(len(open_price_train_trend), len(open_price_train_trend) + len(predicted_sequence)), predicted_sequence, label='Predicted Open Price Trend', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Trend')
    plt.title('Actual vs Predicted Open Price Trend')
    plt.legend()
    plt.savefig('open_price_trend_prediction.png')

    with open('open_price_hmm.pkl', 'wb') as file:
        pickle.dump(open_price_hmm, file)

    np.save('open_price_transition_prob_mat.npy', open_price_hmm.transition_prob_mat)
    np.save('open_price_emission_prob_mat.npy', open_price_hmm.emission_prob_mat)
    np.save('open_price_stationary_dist.npy', open_price_hmm.stationary_dist)

if __name__ == '__main__':
    main()
