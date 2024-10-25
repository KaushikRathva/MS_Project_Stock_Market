import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import torch

data_csv = 'nifty50_data_2020_2024.csv'

class HMM:
    def __init__(self, states, ob_symbols):
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

        # self.transition_prob_mat = np.zeros((self.n_states, self.n_states))
        # self.emission_prob_mat = np.zeros((self.n_states, self.n_ob_symbols))
        # self.stationary_dist = np.zeros(self.n_states)

        # # Initialize transition probabilities
        # for i in range(self.n_states):
        #     self.transition_prob_mat[i] = np.random.dirichlet(np.ones(self.n_states))

        # # Initialize emission probabilities
        # for i in range(self.n_states):
        #     self.emission_prob_mat[i] = np.random.dirichlet(np.ones(self.n_ob_symbols))

        # # Initialize transition probabilities with random values
        # for i in range(self.n_states):
        #     self.transition_prob_mat[i] = np.random.dirichlet(np.ones(self.n_states))

        # # Initialize emission probabilities with random values
        # for i in range(self.n_states):
        #     self.emission_prob_mat[i] = np.random.dirichlet(np.ones(self.n_ob_symbols))

        # # Initialize stationary distribution with random values
        # self.stationary_dist = np.random.dirichlet(np.ones(self.n_states))

        # self.normalize_matrices()

    def normalize_matrices(self):
        self.transition_prob_mat /= np.where(self.transition_prob_mat.sum(axis=1, keepdims=True) == 0, 1, self.transition_prob_mat.sum(axis=1, keepdims=True))
        self.emission_prob_mat /= np.where(self.emission_prob_mat.sum(axis=1, keepdims=True) == 0, 1, self.emission_prob_mat.sum(axis=1, keepdims=True))
        self.stationary_dist /= np.where(self.stationary_dist.sum() == 0, 1, self.stationary_dist.sum())

    def train(self, data_sequence):
        self.update_phase(data_sequence)

    def viterbi_algorithm(self, observation_sequence):
        T = len(observation_sequence)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialization
        delta[0, :] = self.stationary_dist * self.emission_prob_mat[:, observation_sequence[0]]

        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                delta[t, j] = np.max(delta[t-1, :] * self.transition_prob_mat[:, j]) * self.emission_prob_mat[j, observation_sequence[t]]
                psi[t, j] = np.argmax(delta[t-1, :] * self.transition_prob_mat[:, j])

        # Termination
        states_sequence = np.zeros(T, dtype=int)
        states_sequence[-1] = np.argmax(delta[-1, :])

        # Path backtracking
        for t in range(T-2, -1, -1):
            states_sequence[t] = psi[t+1, states_sequence[t+1]]

        return states_sequence
    
    def predict_next_observation(self, observation_sequence):
        latest_state = self.viterbi_algorithm(observation_sequence)[-1]
        next_state = np.argmax(self.transition_prob_mat[latest_state])
        next_observation = np.argmax(self.emission_prob_mat[next_state])
        return next_observation
   
    def update_phase(self, observation_sequence):
        alpha, beta = self.compute_alpha_beta(observation_sequence)
        gamma = self.compute_gamma(alpha, beta)
        xi = self.compute_xi(observation_sequence, alpha, beta)
        print("alpha:",alpha)
        print("::beta:",beta)
        print("::gamma: ", gamma)
        print("::xi: ", xi)
        gamma_sum = gamma.sum(axis=0)
        # print("gamma_sum: ", gamma_sum)
        xi_sum = xi.sum(axis=0)
        # print("xi_sum: ", xi_sum)
        self.update_parameters(gamma ,gamma_sum , xi_sum, observation_sequence)

    def update_parameters(self, gamma, gamma_sum, xi_sum, observation_sequence):
        # Baum-Welch algorithm for updating parameters
        A_new = np.zeros((self.n_states, self.n_states))
        B_new = np.zeros((self.n_states, self.n_ob_symbols))
        lambda_new = np.zeros(self.n_states)



        # Update transition probabilities
        for i in range(self.n_states):
            for j in range(self.n_states):
                A_new[i, j] = xi_sum[i,j]/ gamma_sum[i]

        # Update emission probabilities
        for i in range(self.n_states):
            for k in range(self.n_ob_symbols):
                # print((np.array(observation_sequence) == k))
                B_new[i, k] = np.sum(gamma[:, i] * (np.array(observation_sequence) == k))
            B_new[i] /= gamma_sum[i]

        # Update initial state distribution
        lambda_new = gamma[0]

        self.transition_prob_mat = A_new
        self.emission_prob_mat = B_new
        self.stationary_dist = lambda_new
        # self.normalize_matrices()

    def forward_algo(self, observation_sequence):
        A = torch.tensor(self.transition_prob_mat, device='cuda').clone().detach()
        B = torch.tensor(self.emission_prob_mat, device='cuda').clone().detach()
        lambda_hmm = torch.tensor(self.stationary_dist, device='cuda').clone().detach()

        observation_sequence = observation_sequence.clone().detach()

        T = len(observation_sequence)
        alpha = torch.zeros((T, self.n_states), device='cuda')

        # Initialization
        alpha[0, :] = lambda_hmm * B[:, observation_sequence[0]]

        # Recursion
        for observation in range(1, T):
            for state in range(self.n_states):
                alpha[observation, state] = torch.sum(alpha[observation-1, :] * A[:, state]) * B[state, observation_sequence[observation]]

        # alpha_scaled = torch.zero_like(alpha)
        # scales = torch.zeros(T, device='cuda')

        # alpha_scaled [0,:] = lambda_hmm * B[:, observation_sequence[0]]
        # scales[0] = torch.sum(alpha_scaled[0,:])

#################################################################
        # calculation of alpha scalled
#################################################################
        # Initialize alpha_scaled and scaling factors
        alpha_scaled = torch.zeros((T, self.n_states), device='cuda')
        scales = torch.zeros(T, device='cuda')
        
        # Base case: at time t=0
        alpha_scaled[0, :] = lambda_hmm * B[:, observation_sequence[0]]
        scales[0] = torch.sum(alpha_scaled[0, :])
        if scales[0] > 0:
            alpha_scaled[0, :] /= scales[0]
        
        # Iterate forward in time
        for t in range(1, T):
            for j in range(self.n_states):
                alpha_scaled[t, j] = torch.sum(alpha_scaled[t-1, :] * A[:, j]) * B[j, observation_sequence[t]]
            # Scaling to avoid underflow
            scales[t] = torch.sum(alpha_scaled[t, :])
            if scales[t] > 0:
                alpha_scaled[t, :] /= scales[t]
        
        # return alpha_scaled, scales
        # return alpha_scaled, scales
        return alpha

    def backward_algo(self, observation_sequence, scales=None):
        A = torch.tensor(self.transition_prob_mat, device='cuda').clone().detach()
        B = torch.tensor(self.emission_prob_mat, device='cuda').clone().detach()

        observation_sequence = observation_sequence.clone().detach()

        T = len(observation_sequence)
        beta = torch.zeros((T, self.n_states), device='cuda')

        # Initialization
        beta[-1, :] = 1
        
        # Recursion
        for observation in range(T-2, -1, -1):
            for state in range(self.n_states):
                beta[observation, state] = torch.sum(A[state, :] * B[:, observation_sequence[observation+1]] * beta[observation+1, :])
        beta /= beta.sum(axis=1, keepdim=True)

        # Initialize beta_scaled
        # beta_scaled = torch.zeros(T, self.n_states, device='cuda')

        # # Base case: at time T-1, beta_T-1(i) = 1 for all states, then scale
        # beta_scaled[T-1, :] = 1.0

        # # Iterate backward in time
        # for t in range(T - 2, -1, -1):
        #     for i in range(self.n_states):

        #         beta_scaled[t, i] = torch.sum(A[i, :] * B[:, observation_sequence[t+1]] * beta_scaled[t+1, :])
        #     beta_scaled[t, :] /= scales[t]

        # return beta_scaled
        return beta

    def compute_alpha_beta(self, observation_sequence):
        # print("Computing alpha and beta")

        observation_sequence = torch.tensor(observation_sequence, device='cuda').clone().detach()

        # alpha, scales  = self.forward_algo(observation_sequence)
        # beta = self.backward_algo(observation_sequence, scales)
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
        # print("Computing xi")
        A = self.transition_prob_mat
        B = self.emission_prob_mat
        xi = np.zeros((len(observation_sequence)-1, self.n_states, self.n_states))
        for observation in range(len(observation_sequence)-1):
            # denom = np.sum(alpha[observation, :, None] * A * B[:, observation_sequence[observation+1]] * beta[observation+1, :])
            denom = np.sum(alpha[observation, :, None] * beta[observation, :])
            if denom == 0:
                denom = 1  # Avoid division by zero
            xi[observation] = (alpha[observation, :, None] * A * B[:, observation_sequence[observation+1]] * beta[observation+1, :]) / denom
        return xi

def find_trands(data, threshold):
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
    # train_data = {key: values[:20] for key, values in data_dict.items()}
    # test_data = {key: values[20:40] for key, values in data_dict.items()}
    train_data = {key: values[:100] for key, values in data_dict.items()}
    test_data = {key: values[100:200] for key, values in data_dict.items()}
    # train_data = {key: values[:3] for key, values in data_dict.items()}
    # test_data = {key: values[3:6] for key, values in data_dict.items()}
    return train_data, test_data

# def main():
#     # hmm_model = HMM(['Rainy','Sunnny'],['Walk','Shop'])  #,'Clean'])
#     # hmm_model.transition_prob_mat = np.array([[0.7, 0.3], [0.4, 0.6]])
#     # hmm_model.emission_prob_mat = np.array([[0.5, 0.5], [0.1, 0.9]])

#     # hmm_model.stationary_dist = np.array([0.6, 0.4])
#     # beta = hmm_model.backward_algo([0, 1])
#     # print(beta)
#     states = ('s', 't')
#     possible_observation = ('A','B' )
#     test = HMM(states, possible_observation)
#     test.transition_prob_mat = np.array([[0.6, 0.4], [0.3, 0.7]])
#     test.emission_prob_mat = np.array([[0.3, 0.7], [0.4, 0.6]])
#     test.stationary_dist = np.array([0.5, 0.5])

#     observations = [0, 1, 1, 0]
#     test.backward_algo(observations)
    
def main():
    train_data, test_data = ready_data(data_csv)
    # states = ('strong-negative','negative','neutral','positive','strong-positive')
    states = ('very-strong-negative', 'strong-negative', 'negative', 'neutral', 'positive', 'strong-positive', 'very-strong-positive')
    possible_observation = ('decrease','no-change','increase')
    hmm_model = HMM(states, possible_observation)
    open_price_hmm = cp.deepcopy(hmm_model)
    
    print("Initial Model:")
    print("Transition Probabilities:")
    print(open_price_hmm.transition_prob_mat)
    print("Emission Probabilities:")
    print(open_price_hmm.emission_prob_mat)
    print("Stationary Distribution:")
    print(open_price_hmm.stationary_dist)

    key = 'Open'
    open_price_train_trend = find_trands(train_data[key], 50)
    
    # Train and test on the same data until the model is accurate
    # while 1:
    #     open_price_hmm.train(open_price_train_trend)
    #     predicted_sequence = [open_price_train_trend[0]]
    #     for i in range(1, len(open_price_train_trend)):
    #         next_observation = open_price_hmm.predict_next_observation(predicted_sequence[-1:])
    #         predicted_sequence.append(next_observation)
        
    #     # Check accuracy
    #     accuracy = np.mean(np.array(predicted_sequence) == np.array(open_price_train_trend))
    #     print(f"Iteration accuracy: {accuracy}")
    #     if accuracy >= 0.95:  # Adjust the accuracy threshold as needed
    #         break
    # for i in range(10):
    open_price_hmm.train(open_price_train_trend)

    print("Training done.")

    open_price_test_trend = find_trands(test_data[key], 50)

    last_observation_no = len(open_price_test_trend)
    last_observation = open_price_train_trend[-last_observation_no:]
    predicted_sequence = last_observation.copy()

    for _ in range(last_observation_no):
        predicted_observation = open_price_hmm.predict_next_observation(predicted_sequence)

        # print(predicted_observation)
        predicted_sequence.append(predicted_observation)
        predicted_sequence = predicted_sequence[1:]

    # predicted_sequence = predicted_sequence[-len(open_price_test_trend):]
    # # print(next_observation)
    # predicted_sequence = predicted_sequence[-len(open_price_test_trend):]

    print("Model:")
    print("Transition Probabilities:")
    print(open_price_hmm.transition_prob_mat)
    print("Emission Probabilities:")
    print(open_price_hmm.emission_prob_mat)
    print("Stationary Distribution:")
    print(open_price_hmm.stationary_dist)
    print("Actual Sequence:")
    print(open_price_test_trend)
    print("Predicted Sequence:")
    print(predicted_sequence)
    

    # plt.figure(figsize=(12, 6))
    # plt.subplot(2, 1, 1)
    # plt.plot(open_price_test_trend, label='Actual Open Price Trend')
    # plt.ylabel('Trend')
    # plt.title('Actual Open Price Trend')
    # plt.legend()

    # plt.subplot(2, 1, 2)
    # plt.plot(range(len(open_price_train_trend), len(open_price_train_trend) + len(predicted_sequence)), predicted_sequence, label='Predicted Open Price Trend', linestyle='--')
    # plt.xlabel('Time')
    # plt.ylabel('Trend')
    # plt.title('Predicted Open Price Trend')
    # plt.legend()
    # plt.savefig('open_price_trend_prediction.png')

    plt.figure(figsize=(12, 6))
    plt.plot(open_price_test_trend, label=f'Actual {key} Price Trend')
    plt.plot(range(len(open_price_test_trend)), predicted_sequence, label=f'Predicted {key} Price Trend', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Trend')
    plt.title(f'Actual vs Predicted {key} Price Trend')
    plt.legend()
    
    # Space out x-axis labels
    plt.xticks(ticks=range(0, len(open_price_test_trend), max(1, len(open_price_test_trend) // 10)), rotation=45)
    
    plt.savefig(f'{key}_price_trend_comparison.png')

if __name__ == '__main__':
    main()
