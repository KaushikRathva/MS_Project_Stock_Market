import numpy as np
import matplotlib.pyplot as plt
 
data_csv = 'nifty50_data_2020_2024.csv'
 
class HMM:
    def __init__(self, states, ob_symbols):
        self.states = states
        self.ob_symbols = ob_symbols
        self.n_states = len(states)
        self.n_ob_symbols = len(ob_symbols)
 
        # Initialize transition, emission, and initial state probabilities randomly and normalize
        self.transition_prob_mat = np.random.rand(self.n_states, self.n_states)
        self.transition_prob_mat /= self.transition_prob_mat.sum(axis=1, keepdims=True)
        self.emission_prob_mat = np.random.rand(self.n_states, self.n_ob_symbols)
        self.emission_prob_mat /= self.emission_prob_mat.sum(axis=1, keepdims=True)
        self.stationary_dist = np.random.rand(self.n_states)
        self.stationary_dist /= self.stationary_dist.sum()
 
    def train(self, data_sequence, max_iter=100, tol=1e-4):
        prev_log_likelihood = None
        for iteration in range(max_iter):
            alpha, beta, scales = self.compute_alpha_beta(data_sequence)
            log_likelihood = np.sum(np.log(scales))
            gamma = self.compute_gamma(alpha, beta)
            xi = self.compute_xi(data_sequence, alpha, beta)
            self.update_parameters(gamma, xi, data_sequence)
 
            if prev_log_likelihood is not None:
                if abs(log_likelihood - prev_log_likelihood) < tol:
                    print(f'Converged after {iteration} iterations')
                    break
            prev_log_likelihood = log_likelihood
 
    def compute_alpha_beta(self, observation_sequence):
        T = len(observation_sequence)
        N = self.n_states
        alpha = np.zeros((T, N))
        beta = np.zeros((T, N))
        scales = np.zeros(T)
 
        # Initialization
        alpha[0, :] = self.stationary_dist * self.emission_prob_mat[:, observation_sequence[0]] + 1e-8
        scales[0] = alpha[0, :].sum()
        alpha[0, :] /= scales[0]
 
        # Forward pass
        for t in range(1, T):
            alpha[t, :] = (alpha[t-1, :].dot(self.transition_prob_mat)) * self.emission_prob_mat[:, observation_sequence[t]] + 1e-8
            scales[t] = alpha[t, :].sum()
            alpha[t, :] /= scales[t]
 
        # Backward pass
        beta[-1, :] = 1.0 / scales[-1]
        for t in range(T - 2, -1, -1):
            beta[t, :] = (self.transition_prob_mat.dot((self.emission_prob_mat[:, observation_sequence[t+1]] * beta[t+1, :]))) / scales[t+1]
 
        return alpha, beta, scales
 
    def compute_gamma(self, alpha, beta):
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma
 
    def compute_xi(self, observation_sequence, alpha, beta):
        T = len(observation_sequence)
        N = self.n_states
        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            denominator = (alpha[t, :].reshape(-1, 1) * self.transition_prob_mat * self.emission_prob_mat[:, observation_sequence[t+1]] * beta[t+1, :]).sum()
            xi[t, :, :] = (alpha[t, :].reshape(-1, 1) * self.transition_prob_mat * self.emission_prob_mat[:, observation_sequence[t+1]] * beta[t+1, :]) / denominator
        return xi
 
    def update_parameters(self, gamma, xi, observation_sequence):
        T = len(observation_sequence)
        N = self.n_states
        M = self.n_ob_symbols
        epsilon = 1e-8
 
        # Update initial state distribution
        self.stationary_dist = gamma[0, :] + epsilon
        self.stationary_dist /= self.stationary_dist.sum()
 
        # Update transition probabilities
        for i in range(N):
            for j in range(N):
                numerator = xi[:, i, j].sum()
                denominator = gamma[:-1, i].sum()
                self.transition_prob_mat[i, j] = (numerator + epsilon) / (denominator + N * epsilon)
        self.transition_prob_mat /= self.transition_prob_mat.sum(axis=1, keepdims=True)
 
        # Update emission probabilities
        for i in range(N):
            for k in range(M):
                mask = np.array(observation_sequence) == k
                numerator = gamma[mask, i].sum()
                denominator = gamma[:, i].sum()
                self.emission_prob_mat[i, k] = (numerator + epsilon) / (denominator + M * epsilon)
        self.emission_prob_mat /= self.emission_prob_mat.sum(axis=1, keepdims=True)
 
    def viterbi_algorithm(self, observation_sequence):
        T = len(observation_sequence)
        N = self.n_states
        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)
        epsilon = 1e-8
 
        # Initialization
        delta[0, :] = np.log(self.stationary_dist + epsilon) + np.log(self.emission_prob_mat[:, observation_sequence[0]] + epsilon)
 
        # Recursion
        for t in range(1, T):
            for j in range(N):
                temp = delta[t-1, :] + np.log(self.transition_prob_mat[:, j] + epsilon)
                psi[t, j] = np.argmax(temp)
                delta[t, j] = np.max(temp) + np.log(self.emission_prob_mat[j, observation_sequence[t]] + epsilon)
 
        # Termination
        states_sequence = np.zeros(T, dtype=int)
        states_sequence[-1] = np.argmax(delta[-1, :])
 
        # Path backtracking
        for t in range(T - 2, -1, -1):
            states_sequence[t] = psi[t + 1, states_sequence[t + 1]]
 
        return states_sequence
 
    def predict_next_observation(self, observation_sequence):
        state_sequence = self.viterbi_algorithm(observation_sequence)
        latest_state = state_sequence[-1]
        next_state_probs = self.transition_prob_mat[latest_state, :]
        next_state = np.argmax(next_state_probs)
        next_observation_probs = self.emission_prob_mat[next_state, :]
        next_observation = np.argmax(next_observation_probs)
        return next_observation
 
def find_trends(data, threshold):
    trends = []
    for i in range(len(data) - 1):
        diff = data[i+1] - data[i]
        if abs(diff) < threshold:
            trends.append(1)  # no-change
        elif diff < 0:
            trends.append(0)  # decrease
        else:
            trends.append(2)  # increase
    return trends
 
def ready_data(data_csv):
    raw_data = np.genfromtxt(data_csv, delimiter=',', dtype=None, encoding=None, skip_header=1)
    with open(data_csv, 'r') as f:
        header = f.readline().strip().split(',')
 
    data_dict = {key: [] for key in header}
    for row in raw_data:
        for key, value in zip(header, row):
            if key == 'Date':
                data_dict[key].append(np.datetime64(value))
            else:
                try:
                    data_dict[key].append(float(value))
                except ValueError:
                    data_dict[key].append(np.nan)
 
    # Convert lists to numpy arrays
    for key in data_dict:
        data_dict[key] = np.array(data_dict[key])
 
    # Remove entries with NaN values
    valid_indices = ~np.isnan(data_dict['Open'])
    for key in data_dict:
        data_dict[key] = data_dict[key][valid_indices]
 
    if len(data_dict['Open']) < 320:
        raise ValueError("Not enough data points for the specified train and test split.")
 
    # Use the last 310 data points for training and last 10 for testing
    train_data = {key: values[-320:-10] for key, values in data_dict.items()}
    test_data = {key: values[-10:] for key, values in data_dict.items()}
    return train_data, test_data
 
def main():
    train_data, test_data = ready_data(data_csv)
    key = 'Low'
    train_prices = train_data[key]
    test_prices = test_data[key]
 
    # Define observation symbols
    possible_observations = [0, 1, 2]  # decrease, no-change, increase
 
    # Get trends
    train_trends = find_trends(train_prices, 25)
    observations = train_trends
 
    # Define states as indices
    states = list(range(5))  # Adjusted to 5 states for simplicity
 
    hmm_model = HMM(states, possible_observations)
    hmm_model.train(observations)
 
    print("Training done.")
 
    test_trends = find_trends(test_prices, 25)
    test_observations = test_trends
 
    # Use the last few observations from training as context
    context_length = 5
    context_observations = observations[-context_length:]
 
    predicted_observations = []
    for obs in test_observations:
        predicted_observation = hmm_model.predict_next_observation(context_observations)
        predicted_observations.append(predicted_observation)
        context_observations.append(obs)
        context_observations = context_observations[1:]
 
    plt.figure(figsize=(12, 6))
    plt.plot(test_observations, label=f'Actual {key} Price Trend')
    plt.plot(predicted_observations, label=f'Predicted {key} Price Trend', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Trend')
    plt.title(f'Actual vs Predicted {key} Price Trend')
    plt.legend()
    plt.savefig(f'{key}_price_trend_comparison.png')
 
if __name__ == '__main__':
    main()
 
 