import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import torch
 
data_csv = 'nifty50_data_2020_2024.csv'
 
class HMM:
    def __init__(self, states, ob_symbols):
    # def __init__(self, n_states, n_ob_symbols):
        self.states = states
        self.ob_symbols = ob_symbols
        self.n_states = len(states)
        self.n_ob_symbols = len(ob_symbols)

        # Initialize transition, emission, and initial state probabilities
        self.transition_prob_mat = np.random.rand(self.n_states, self.n_states)
        self.transition_prob_mat /= self.transition_prob_mat.sum(axis=1, keepdims=True)
        self.emission_prob_mat = np.random.rand(self.n_states, self.n_ob_symbols)
        self.emission_prob_mat /= self.emission_prob_mat.sum(axis=1, keepdims=True)
        self.stationary_dist = np.random.rand(self.n_states)
        self.stationary_dist /= self.stationary_dist.sum()
        self.normalize_matrices()
        self.load_model()

    def load_model(self):
        try:
            self.transition_prob_mat = np.load('transition_prob_mat.npy')
            self.emission_prob_mat = np.load('emission_prob_mat.npy')
            self.stationary_dist = np.load('stationary_dist.npy')
        except FileNotFoundError:
            print("Model files not found. Using randomly initialized matrices.")

 
    def normalize_matrices(self):
        self.transition_prob_mat /= np.where(self.transition_prob_mat.sum(axis=1, keepdims=True) == 0, 1, self.transition_prob_mat.sum(axis=1, keepdims=True))
        self.emission_prob_mat /= np.where(self.emission_prob_mat.sum(axis=1, keepdims=True) == 0, 1, self.emission_prob_mat.sum(axis=1, keepdims=True))
        self.stationary_dist /= np.where(self.stationary_dist.sum() == 0, 1, self.stationary_dist.sum())
 
    def train(self, data_sequence):
        self.update_phase(data_sequence)
 
    # def viterbi_algorithm(self, observation_sequence):
    #     T = len(observation_sequence)
    #     V = np.zeros((T, self.n_states))
    #     path = np.zeros((T, self.n_states), dtype=int)
 
    #     for s in range(self.n_states):
    #         V[0, s] = self.stationary_dist[s] * self.emission_prob_mat[s, int(observation_sequence[0])]
 
    #     # Recursion
    #     for t in range(1, T-1):
    #         for s in range(self.n_states):
    #             prob = V[t-1, :] * self.transition_prob_mat[:, s] * self.emission_prob_mat[s, int(observation_sequence[t])]
    #             V[t, s] = np.max(prob)
    #             path[t, s] = np.argmax(prob)
 
    #     # Termination
    #     best_path_prob = np.max(V[T-1, :])
    #     best_last_state = np.argmax(V[T-1, :])
 
    #     # Path backtracking
    #     best_path = np.zeros(T, dtype=int)
    #     best_path[-1] = best_last_state
    #     for t in range(T-2, -1, -1):
    #         best_path[t] = path[t+1, best_path[t+1]]
 
    #     return best_path
    
    def viterbi_algorithm(self, observation_sequence):
        T = len(observation_sequence)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialization
        delta[0, :] = self.stationary_dist * self.emission_prob_mat[:, observation_sequence[0]]

        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                observation_index = self.ob_symbols.index(observation_sequence[t])
                delta[t, j] = np.max(delta[t-1, :] * self.transition_prob_mat[:, j]) * self.emission_prob_mat[j, observation_index]
                psi[t, j] = np.argmax(delta[t-1, :] * self.transition_prob_mat[:, j])

        # Termination
        states_sequence = np.zeros(T, dtype=int)
        states_sequence[-1] = np.argmax(delta[-1, :])

        # Path backtracking
        for t in range(T-2, -1, -1):
            states_sequence[t] = psi[t+1, states_sequence[t+1]]

        return states_sequence
    
    # def predict_next_observation(self, observation_sequence):
    #     latest_state = self.viterbi_algorithm(observation_sequence)[-1]
    #     next_state = np.argmax(self.transition_prob_mat[latest_state])
    #     next_observation = np.argmax(self.emission_prob_mat[next_state])
    #     # print("latest_state: ", latest_state)
    #     # print(",next_state: ", next_state)
    #     # print(",next_observation: ", next_observation)
    #     # print()
    #     return next_observation
    def predict_next_observation(self, observation_sequence):
        latest_state = self.viterbi_algorithm(observation_sequence)[-1]
        next_observation_prob = np.zeros(self.n_ob_symbols)
        for observation in range(self.n_ob_symbols):
            next_observation_prob[observation] = np.sum(self.transition_prob_mat[latest_state] * self.emission_prob_mat[:, observation])
        return np.argmax(next_observation_prob)
    
    def update_phase(self, observation_sequence):
        alpha, beta, scales = self.compute_alpha_beta(observation_sequence)
        gamma = self.compute_gamma(alpha, beta, scales)
        xi = self.compute_xi(observation_sequence, alpha, beta, scales)
        gamma_sum = gamma.sum(axis=0)

        xi_sum = xi.sum(axis=0)
        self.update_parameters(gamma, gamma_sum, xi_sum, observation_sequence)
 
    def update_parameters(self, gamma, gamma_sum, xi_sum, observation_sequence):
        # Baum-Welch algorithm for updating parameters
        A_new = np.zeros((self.n_states, self.n_states))
        B_new = np.zeros((self.n_states, self.n_ob_symbols))
        lambda_new = np.zeros(self.n_states)
 
        # Update transition probabilities
        for i in range(self.n_states):
            for j in range(self.n_states):
                xi_sum_k = np.zeros(self.n_states)
                for k in range(self.n_states):
                    xi_sum_k += xi_sum[i, k]
                A_new[i, j] = xi_sum[i, j]/xi_sum_k[i]
                # A_new[i, j] = xi_sum[i, j] 
                # A_new[i, j] = xi_sum[i, j] / 
 
        # Update emission probabilities
        for i in range(self.n_states):
            for k in range(self.n_ob_symbols):
                B_new[i, k] = np.sum(gamma[:, i] * (np.array(observation_sequence) == k))
            B_new[i] /= gamma_sum[i]
 
        # Update initial state distribution
        lambda_new = gamma[0]
 
        self.transition_prob_mat = A_new
        self.emission_prob_mat = B_new
        self.stationary_dist = lambda_new
 
    def forward_algo(self, observation_sequence):
        A = torch.tensor(self.transition_prob_mat, device='cuda').clone().detach()
        B = torch.tensor(self.emission_prob_mat, device='cuda').clone().detach()
        lambda_hmm = torch.tensor(self.stationary_dist, device='cuda').clone().detach()
 
        observation_sequence = observation_sequence.clone().detach()
 
        T = len(observation_sequence)
        alpha_scaled = torch.zeros((T, self.n_states), device='cuda')
        scales = torch.zeros(T, device='cuda')
        # Initialization
        alpha_scaled[0, :] = lambda_hmm * B[:, observation_sequence[0]]
        scales[0] = torch.sum(alpha_scaled[0, :])
        if scales[0] > 0:
            alpha_scaled[0, :] /= scales[0]
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                
                alpha_scaled[t, j] = torch.sum(alpha_scaled[t-1, :] * A[:, j]) * B[j, observation_sequence[t]]
            scales[t] = torch.sum(alpha_scaled[t, :])
            if scales[t] > 0:
                alpha_scaled[t, :] /= scales[t]
        return alpha_scaled, scales
 
    def backward_algo(self, observation_sequence, scales):
        A = torch.tensor(self.transition_prob_mat, device='cuda').clone().detach()
        B = torch.tensor(self.emission_prob_mat, device='cuda').clone().detach()
 
        observation_sequence = observation_sequence.clone().detach()
 
        T = len(observation_sequence)
        beta_scaled = torch.zeros((T, self.n_states), device='cuda')
 
        # Initialization
        beta_scaled[-1, :] = 1.0
 
        # Recursion
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                beta_scaled[t, i] = torch.sum(A[i, :] * B[:, observation_sequence[t+1]] * beta_scaled[t+1, :])
            beta_scaled[t, :] /= scales[t+1]
 
        return beta_scaled
 
    def compute_alpha_beta(self, observation_sequence):
        observation_sequence = torch.tensor(observation_sequence, device='cuda').clone().detach()
 
        alpha, scales = self.forward_algo(observation_sequence)
        beta = self.backward_algo(observation_sequence, scales)
 
        alpha = alpha.cpu().numpy()
        beta = beta.cpu().numpy()
        scales = scales.cpu().numpy()
 
        return alpha, beta, scales
 
    def compute_gamma(self, alpha, beta, scales):
        gamma = alpha * beta
        # sum_gamma = gamma.sum(axis=1, keepdims=True)
        for state_i in range(self.n_states):
            for t in range(self.n_ob_symbols):
                gamma[t, state_i] /= scales[t]
        return gamma
 
    def compute_xi(self, observation_sequence, alpha, beta, scales):
        xi = np.zeros((len(observation_sequence)-1, self.n_states, self.n_states))
        A = self.transition_prob_mat
        B = self.emission_prob_mat
        for t in range(len(observation_sequence)-1):
            o_next = observation_sequence[t+1]
            denom = scales[t+1]  # Scaling factor at time t+1
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = alpha[t, i] * A[i, j] * B[j, o_next] * beta[t+1, j]
            # xi[t, :, :] /= denom
            xi_sum = xi[t].sum()
            if xi_sum > 0:
                xi[t] /= xi_sum
        return xi
 
def find_trands(data, threshold):
    # mean = np.mean(data)
    # standard_deviation = np.std(data)
    # print(mean)
    # print(standard_deviation)
    
    trends = []
    observations = [-1,0,1]
    # observations = ['decrease','no-change','increase']
    # observations = [0,1,2]
    for i in range(len(data) - 1):
        dif = data[i+1] - data[i]
        if abs(dif) < threshold:
            trends.append(observations[1])
        elif dif < 0:
            trends.append(observations[0])
        else:
            trends.append(observations[2])
    # for i in range(len(data)):
    #     trends.append(round((data[i] - mean) / standard_deviation))

    # print(trends)
    return trends,observations
 
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
 
    # train_data = {key: values[:len(values)//2] for key, values in data_dict.items()}
    # test_data = {key: values[len(values)//2:] for key, values in data_dict.items()}
    # Ensure there are enough data points
    if len(data_dict['Open']) < 220:
        raise ValueError("Not enough data points for the specified train and test split.")


    train_len = 300
    test_len = 20
    # Randomly select a starting point for the consecutive chunk
    start_index = np.random.randint(0, len(data_dict['Open']) - train_len - test_len)



    train_data = {key: values[start_index:start_index + train_len] for key, values in data_dict.items()}
    test_data = {key: values[(start_index + train_len + 1):(start_index + train_len + test_len +1)] for key, values in data_dict.items()}
    return train_data, test_data
 
def main():
    train_data, test_data = ready_data(data_csv)
    key = 'Low'
    train_data = train_data[key]
    test_data = test_data[key]
    threshold = 100
    
    train_trend,possible_observation = find_trands(train_data, threshold)
    test_trend,possible_observation = find_trands(test_data, threshold)

    # states = ('strong-negative', 'negative', 'neutral', 'positive', 'strong-positive')
    states = ('dont-buy', 'very-strong-negative', 'strong-negative', 'negative', 'neutral', 'positive', 'strong-positive', 'very-strong-positive', 'buy-right-now')
    # possible_observation = ('decrease', 'no-change', 'increase')
    hmm_model = HMM(states, possible_observation)
    # hmm_model = HMM(7,len(possible_observ))
    hmm = cp.deepcopy(hmm_model)

    print("Initial Model:")
    print("Transition Probabilities:")
    print(hmm.transition_prob_mat)
    print("Emission Probabilities:")
    print(hmm.emission_prob_mat)
    print("Stationary Distribution:")
    print(hmm.stationary_dist)
    
    hmm.train(train_trend)
    print("Training done.")
 
    context_length = 10
    context_observations = train_trend[-context_length:]
 
    predicted_observations = []
    for obs in test_trend:
        predicted_observation = hmm_model.predict_next_observation(context_observations)
        predicted_observations.append(predicted_observation)
        context_observations.append(obs)
        context_observations = context_observations[1:]
 
    plt.figure(figsize=(12, 6))
    plt.plot(test_trend, label=f'Actual {key} Price Trend')
    plt.plot(predicted_observations, label=f'Predicted {key} Price Trend', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Trend')
    plt.title(f'Actual vs Predicted {key} Price Trend')
    plt.legend()

    # Save the model parameters
    np.save('transition_prob_mat.npy', hmm.transition_prob_mat)
    np.save('emission_prob_mat.npy', hmm.emission_prob_mat)
    np.save('stationary_dist.npy', hmm.stationary_dist)
 
    # # last_observation_no = len(open_price_test_trend)
    # initial_observation_lenght = 10
    # last_observation = open_price_train_trend[-initial_observation_lenght:]
    # predicted_sequence = last_observation.copy()
    
    # for i in range(len(open_price_test_trend)):
    #     predicted_observation = open_price_hmm.predict_next_observation(predicted_sequence)
    #     print(predicted_observation)
    #     predicted_sequence.append(predicted_observation)
    #     if i > len(open_price_test_trend) - initial_observation_lenght: 
    #         predicted_sequence = predicted_sequence[1:]

    # plt.figure(figsize=(12, 6))
    # plt.plot(open_price_test_trend, label=f'Actual {key} Price Trend')
    # plt.plot(range(len(open_price_test_trend)), predicted_sequence, label=f'Predicted {key} Price Trend', linestyle='--')
    # plt.xlabel('Time')
    # plt.ylabel('Trend')
    # plt.title(f'Actual vs Predicted {key} Price Trend')
    # plt.legend()
    plt.savefig(f'{key}_price_trend_comparison.png')
 
if __name__ == '__main__':
    main()




# for _ in range(len(open_price_test_trend)):
#     best_path = open_price_hmm.veterbi_algorithm(predicted_sequence)
#     next_observation = best_path[-1]
#     predicted_sequence.append(next_observation)
#     predicted_sequence = predicted_sequence[1:]

# print("Model:")
# print("Transition Probabilities:")
# print(open_price_hmm.transition_prob_mat)
# print("Emission Probabilities:")
# print(open_price_hmm.emission_prob_mat)
# print("Stationary Distribution:")
# print(open_price_hmm.stationary_dist)
# print("Actual Sequence:")
# print(open_price_test_trend)
# print("Predicted Sequence:")
# print(predicted_sequence)