import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import torch

data_csv = 'nifty50_data_2020_2024.csv'

class HMM:
    def __init__(self, n_states, n_symbols):
        self.n_states = n_states
        self.n_symbols = n_symbols
        # self.transition_prob_mat = np.zeros((n_states, n_states))
        # self.emission_prob_mat = np.zeros((n_states, n_symbols))
        # self.stationary_dist = np.zeros(n_states)
        self.transition_prob_mat = np.random.rand(n_states, n_states)
        self.transition_prob_mat /= self.transition_prob_mat.sum(axis=1, keepdims=True)
        
        self.emission_prob_mat = np.random.rand(n_states, n_symbols)
        self.emission_prob_mat /= self.emission_prob_mat.sum(axis=1, keepdims=True)
        
        self.stationary_dist = np.random.rand(n_states)
        self.stationary_dist /= self.stationary_dist.sum()
        self.normalize_matrices()

    def normalize_matrices(self):
        self.transition_prob_mat /= np.where(self.transition_prob_mat.sum(axis=1, keepdims=True) == 0, 1, self.transition_prob_mat.sum(axis=1, keepdims=True))
        self.emission_prob_mat /= np.where(self.emission_prob_mat.sum(axis=1, keepdims=True) == 0, 1, self.emission_prob_mat.sum(axis=1, keepdims=True))
        self.stationary_dist /= np.where(self.stationary_dist.sum() == 0, 1, self.stationary_dist.sum())

    def train(self, data_sequence):
        self.update_phase(data_sequence)

    def predict_next_observation(self, observation_sequence):
        A = torch.tensor(self.transition_prob_mat, device='cuda').clone().detach()
        B = torch.tensor(self.emission_prob_mat, device='cuda').clone().detach()
        lambda_hmm = torch.tensor(self.stationary_dist, device='cuda').clone().detach()

        observation_sequence = torch.tensor(observation_sequence, device='cuda').clone().detach()

        T = len(observation_sequence)
        delta = torch.zeros((T, self.n_states), device='cuda')

        # Initialization
        delta[0, :] = lambda_hmm * B[:, observation_sequence[0]]

        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                delta[t, j] = torch.sum(delta[t-1, :] * A[:, j]) * B[j, observation_sequence[t]]

        # Predict the next observation
        next_observation_prob = torch.zeros(self.n_symbols, device='cuda')
        for j in range(self.n_states):
            next_observation_prob += delta[T-1, j] * B[j, :]
        next_observation = torch.argmax(next_observation_prob).item()

        return next_observation
   
    def update_phase(self, observation_sequence):
        alpha, beta = self.compute_alpha_beta(observation_sequence)
        gamma = self.compute_gamma(alpha, beta)
        xi = self.compute_xi(observation_sequence, alpha, beta)
        print("alpha:",alpha)
        print("::beta:",beta)
        print("::gamma: ", gamma)
        print("::xi: ", xi)
        self.update_parameters(gamma, xi, observation_sequence)

    def update_parameters(self, gamma, xi, observation_sequence):
        print("Updating parameters")
        A_new = np.zeros((self.n_states, self.n_states))
        B_new = np.zeros((self.n_states, self.n_symbols))
        lambda_new = np.zeros(self.n_states)

        gamma_sum = gamma.sum(axis=0)
        xi_sum = xi.sum(axis=0)

        A_new = xi_sum / np.where(gamma_sum[:, None] == 0, 1, gamma_sum[:, None])
        for state in range(self.n_states):
            for symbol in range(self.n_symbols):
                mask = np.array(observation_sequence) == symbol
                B_new[state, symbol] = np.sum(gamma[mask, state])
            B_new[state] /= np.where(gamma_sum[state] == 0, 1, gamma_sum[state])

        lambda_new = gamma[0]

        self.transition_prob_mat = A_new
        self.emission_prob_mat = B_new
        self.stationary_dist = lambda_new
        # self.normalize_matrices()

    def forward_algo(self, observation_sequence):
        A = torch.tensor(self.transition_prob_mat, device='cuda').clone().detach()
        B = torch.tensor(self.emission_prob_mat, device='cuda').clone().detach()
        lambda_hmm = torch.tensor(self.stationary_dist, device='cuda').clone().detach()

        observation_sequence = observation_sequence.clone().detach().to('cuda')

        T = len(observation_sequence)
        alpha = torch.zeros((T, self.n_states), device='cuda')

        # Initialization
        for stages in range(self.n_states):
            alpha[0, stages] = lambda_hmm[stages] * B[stages, observation_sequence[0]]

        # Recursion
        for observation in range(T-1):
            for state in range(self.n_states):
                # for prev_state in range(self.n_states):
                #     alpha[observ_seq_len, state] += alpha[observ_seq_len-1, prev_state] * A[prev_state, state] *B[state, observation_sequence[observ_seq_len]]
                alpha[observation, state] = torch.sum(alpha[observation-1, :] * A[:, state]) * B[state, observation_sequence[observation]]

        return alpha

    def backward_algo(self, observation_sequence):
        A = torch.tensor(self.transition_prob_mat, device='cuda').clone().detach()
        B = torch.tensor(self.emission_prob_mat, device='cuda').clone().detach()

        observation_sequence = observation_sequence.clone().detach().to('cuda')

        T = len(observation_sequence)
        beta = torch.zeros((T, self.n_states), device='cuda')

        # Initialization
        beta[-1, :] = 1

        # Recursion
        for observation in range(T-1):
            for state in range(self.n_states):
                beta[observation, state] = torch.sum(A[state, :] * B[:, observation_sequence[observation+1]] * beta[observation+1, :])
        return beta

    def compute_alpha_beta(self, observation_sequence):
        print("Computing alpha and beta")

        observation_sequence = torch.tensor(observation_sequence, device='cuda').clone().detach()

        alpha = self.forward_algo(observation_sequence)
        beta = self.backward_algo(observation_sequence)

        alpha = alpha.cpu().numpy()
        beta = beta.cpu().numpy()

        return alpha, beta

    def compute_gamma(self, alpha, beta):
        gamma = alpha * beta
        sum_gamma = gamma.sum(axis=1, keepdims=True)
        sum_gamma[sum_gamma == 0] = 1  # Avoid division by zero
        gamma /= sum_gamma
        return gamma

    def compute_xi(self, observation_sequence, alpha, beta):
        print("Computing xi")
        A = self.transition_prob_mat
        B = self.emission_prob_mat
        xi = np.zeros((len(observation_sequence)-1, self.n_states, self.n_states))
        for t in range(len(observation_sequence)-1):
            denom = np.sum(alpha[t, :, None] * A * B[:, observation_sequence[t+1]] * beta[t+1, :])
            if denom == 0:
                denom = 1  # Avoid division by zero
            xi[t] = (alpha[t, :, None] * A * B[:, observation_sequence[t+1]] * beta[t+1, :]) / denom
        return xi

def find_trands(data, threshold):
    trends = []
    for i in range(len(data) - 1):
        dif = data[i+1] - data[i]
        if abs(dif) < threshold:
            trends.append(0)
        elif dif < 0:
            trends.append(-1)
        else:
            trends.append(1)
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

    # assert len(data_dict[header[0]]) >= 200, "Not enough data to split into 100 rows each for train and test."

    # train_data = {key: values[:len(values)//2] for key, values in data_dict.items()}
    # test_data = {key: values[len(values)//2:] for key, values in data_dict.items()}
    train_data = {key: values[:20] for key, values in data_dict.items()}
    test_data = {key: values[20:40] for key, values in data_dict.items()}
    # train_data = {key: values[:10] for key, values in data_dict.items()}
    # test_data = {key: values[10:20] for key, values in data_dict.items()}
    return train_data, test_data

def main():
    hmm_model = HMM(2,4)
    hmm_model.transition_prob_mat = np.array([[0.8, 0.2], [0.4, 0.6]])
    hmm_model.emission_prob_mat = np.array([[0.4, 0.1, 0.2, 0.3], [0.3, 0.45, 0.2, 0.05]])

    hmm_model.stationary_dist = np.array([0.5, 0.5])
    hmm_model.backward_algo([])
# def main():
#     train_data, test_data = ready_data(data_csv)
#     hmm_model = HMM(5, 3)
#     open_price_hmm = cp.deepcopy(hmm_model)
    
#     print("Initial Model:")
#     print("Transition Probabilities:")
#     print(open_price_hmm.transition_prob_mat)
#     print("Emission Probabilities:")
#     print(open_price_hmm.emission_prob_mat)
#     print("Stationary Distribution:")
#     print(open_price_hmm.stationary_dist)

#     open_price_train_trend = find_trands(train_data['Open'], 50)
    
#     # Train and test on the same data until the model is accurate
#     # while 1:
#     #     open_price_hmm.train(open_price_train_trend)
#     #     predicted_sequence = [open_price_train_trend[0]]
#     #     for i in range(1, len(open_price_train_trend)):
#     #         next_observation = open_price_hmm.predict_next_observation(predicted_sequence[-1:])
#     #         predicted_sequence.append(next_observation)
        
#     #     # Check accuracy
#     #     accuracy = np.mean(np.array(predicted_sequence) == np.array(open_price_train_trend))
#     #     print(f"Iteration accuracy: {accuracy}")
#     #     if accuracy >= 0.95:  # Adjust the accuracy threshold as needed
#     #         break
    
#     open_price_hmm.train(open_price_train_trend)
#     print("Training done.")

#     open_price_test_trend = find_trands(test_data['Open'], 50)

#     last_observation = open_price_train_trend[-1]
#     predicted_sequence = [last_observation]

#     for _ in range(len(open_price_test_trend)):
#         next_observation = open_price_hmm.predict_next_observation(predicted_sequence[-1:])
#         predicted_sequence.append(next_observation)

#     print("Model:")
#     print("Transition Probabilities:")
#     print(open_price_hmm.transition_prob_mat)
#     print("Emission Probabilities:")
#     print(open_price_hmm.emission_prob_mat)
#     print("Stationary Distribution:")
#     print(open_price_hmm.stationary_dist)
#     print("Actual Sequence:")
#     print(open_price_test_trend)
#     print("Predicted Sequence:")
#     print(predicted_sequence)
    

#     plt.figure(figsize=(12, 6))
#     plt.subplot(2, 1, 1)
#     plt.plot(open_price_test_trend, label='Actual Open Price Trend')
#     plt.ylabel('Trend')
#     plt.title('Actual Open Price Trend')
#     plt.legend()

#     plt.subplot(2, 1, 2)
#     plt.plot(range(len(open_price_train_trend), len(open_price_train_trend) + len(predicted_sequence)), predicted_sequence, label='Predicted Open Price Trend', linestyle='--')
#     plt.xlabel('Time')
#     plt.ylabel('Trend')
#     plt.title('Predicted Open Price Trend')
#     plt.legend()
#     plt.savefig('open_price_trend_prediction.png')

if __name__ == '__main__':
    main()
