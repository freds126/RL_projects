# Close environment# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

# Load packages
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle
from tqdm import trange

"""

This is an implementation of a SARSA(lambda) algo, with fourier basis of order p = 2 - solves the MountainCar problem!

Made by: Fredrick Carlsåker and Hampus Hallberg


To train model and check if it passed, set normal_run = True
    If you also want 3d plots of the value function and policy, set plot_3d_valueFunc = True

You can choos different exploration strategies by changing it in exploration_strategy
    The choices are {epsilon_greedy, 
                    ucb,
                    softmax 
                    }
                    
You can also change a bunch of parameters below


To plot over lambdas, set plot_over_lambda = True

To plot over alphas, set plot_over_alpha = True

"""

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
low, high = env.observation_space.low, env.observation_space.high

# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x

def softmax_exploration(Q_s, temperature=0.5):
    """
    Strategy 2: Softmax (Boltzmann) Exploration
    Select actions probabilistically based on Q-values
    """
    Q_scaled = (Q_s - np.max(Q_s)) / temperature
    exp_Q = np.exp(Q_scaled)
    probs = exp_Q / np.sum(exp_Q)
    action = np.random.choice(len(Q_s), p=probs)
    return action

def ucb_exploration(Q_s, state, action_counts, total_steps, bonus_scale=2.0):
    """
    Strategy 3: Upper Confidence Bound Exploration
    Add exploration bonus to less-visited actions
    """
    state_bin = tuple(np.round(state, 1))
    
    Q_explore = np.zeros_like(Q_s)
    for a in range(len(Q_s)):
        count_a = action_counts.get((state_bin, a), 0) + 1
        bonus = bonus_scale * np.sqrt(np.log(total_steps + 1) / count_a)
        Q_explore[a] = Q_s[a] + bonus
    
    return np.argmax(Q_explore)

def epsilon_greedy(Q_s, epsilon):
    """
    Baseline: Standard epsilon-greedy
    """
    
    if np.random.random() < epsilon:
        return np.random.randint(len(Q_s))
    else:
        return np.argmax(Q_s)

def evaluate():
    # Copyright [2025] [KTH Royal Institute of Technology] 
    # Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
    # This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

    # Load packages


    # Import and initialize Mountain Car Environment
    env = gym.make('MountainCar-v0')
    env.reset()
    k = env.action_space.n      # tells you the number of actions
    low, high = env.observation_space.low, env.observation_space.high


    def scale_state_varibles(s, eta, low=env.observation_space.low, high=env.observation_space.high):
        ''' Rescaling of s to the box [0,1]^2
            and features transformation
        '''
        x = (s-low) / (high-low)
        return np.cos(np.pi * np.dot(eta, x))

    def Qvalues(s, w):
        ''' Q Value computation '''
        return np.dot(w, s)

    # Parameters
    N_EPISODES = 50            # Number of episodes to run for trainings
    CONFIDENCE_PASS = -135

    # Fourier basis
    p = 3
    try:
        f = open('weights.pkl', 'rb')
        data = pickle.load(f)
        if 'W' not in data or 'N' not in data:
            print('Matrix W (or N) is missing in the dictionary.')
            exit(-1)
        w = data['W']
        eta = data['N']

        # Dimensionality checks
        if w.shape[1] != eta.shape[0]:
            print('m is not the same for the matrices W and N')
            exit(-1)
        m = w.shape[1]
        if w.shape[0] != k:
            print('The first dimension of W is not {}'.format(k))
            exit(-1)
        if eta.shape[1] != 2:
            print('The second dimension of eta is not {}'.format(2))
            exit(-1)
    except:
        print('File weights.pkl not found!')
        exit(-1)



    # Reward
    episode_reward_list = []  # Used to store episodes reward


    # Simulate episodes
    print('Checking solution...')
    EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
    for i in EPISODES:
        EPISODES.set_description("Episode {}".format(i))
        # Reset enviroment data
        done = False
        truncated = False
        state = scale_state_varibles(env.reset()[0], eta, low, high)
        total_episode_reward = 0.

        qvalues = Qvalues(state, w)
        action = np.argmax(qvalues)

        while not (done or truncated):
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = scale_state_varibles(next_state, eta, low, high)
            qvalues_next = Qvalues(next_state, w)
            next_action = np.argmax(qvalues_next)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            qvalues = qvalues_next
            action = next_action

        # Append episode reward
        episode_reward_list.append(total_episode_reward)

        # Close environment
        env.close()

    avg_reward = np.mean(episode_reward_list)
    confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)


    print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
                    avg_reward,
                    confidence))

    if avg_reward - confidence >= CONFIDENCE_PASS:
        print('Your policy passed the test!')
    else:
        print('Your policy did not pass the test! The average reward of your policy needs to be greater than {} with 95% confidence'.format(CONFIDENCE_PASS))

    return avg_reward, confidence

def initialize_eta(p: int, include_bias: bool):
    eta = np.array([[i, j] for i in range(p+1) for j in range(p+1)])

    if not include_bias:
        eta = eta[1:, :]
    return eta

def initialize_weights(N: int, K: int):
    # initialise weights and phi
    W = np.zeros((K, N)) 
    return W

def initialize_alphas(eta, alpha_start):
    # initialize alpha w.r.t eta
    alpha = np.ones((len(eta),)) * alpha_start
    L2_norm = np.linalg.norm(eta, axis=1)
    return alpha/(L2_norm + (L2_norm == 0))  

def calculate_phi(s):
    # s is unscaled (env state)
    s_scaled = scale_state_variables(s)
    return np.cos(np.pi * (eta @ s_scaled))  # shape (9,)

def calculate_Q_values(s):
    # returns Q(s,a) for all actions as a vector of length 3
    phi_s = calculate_phi(s)
    return W @ phi_s  # shape (3,)

def save_W_eta(data, filename="weights.pkl"):
    with open("weights.pkl", "wb") as f:
        pickle.dump(data, f)

def plot_3d_valueFunc_policy(N_episodes, episode_reward_list, save=False):
    
    # Plot Rewards
    plt.plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
    plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 10), label='Average episode reward')
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.title('Total Reward vs Episodes')
    plt.legend()
    plt.grid(alpha=0.3)
    if save:
        plt.savefig("reward-episodes.png")
    plt.show()

    from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots

    # State space ranges
    pos_min, vel_min = env.observation_space.low
    pos_max, vel_max = env.observation_space.high

    # Grid resolution
    n_pos = 50
    n_vel = 50

    # Grids
    pos_grid = np.linspace(pos_min, pos_max, n_pos)
    vel_grid = np.linspace(vel_min, vel_max, n_vel)

    # Correct meshgrid (X=pos, Y=vel)
    P, V = np.meshgrid(pos_grid, vel_grid, indexing='xy')

    V_values = np.zeros((n_vel, n_pos))

    for i in range(n_vel):     # velocity index
        for j in range(n_pos): # position index
            s = np.array([P[i,j], V[i,j]])   # correct state order
            Qs = calculate_Q_values(s)
            V_values[i, j] = np.max(Qs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(P, V, V_values, linewidth=0, antialiased=True)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('V(s) = max_a Q(s,a)')
    ax.set_title('Approximate Value Function of Learned Policy')
    if save:
        plt.savefig("Value_func.png")
    plt.show()

    policy = np.zeros((n_vel, n_pos))

    for i in range(n_vel):
        for j in range(n_pos):
            s = np.array([P[i,j], V[i,j]])
            Qs = calculate_Q_values(s)
            policy[i,j] = np.argmax(Qs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(P, V, policy, linewidth=0, antialiased=True)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Action (0=left,1=none,2=right)')
    ax.set_title('Optimal Policy over State Space')
    if save:
        plt.savefig("3d-policy.png")
    plt.show()

    plt.figure()
    plt.pcolormesh(P, V, policy, shading='auto')
    cbar = plt.colorbar()
    cbar.set_label('Action (0=left,1=none,2=right)')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Optimal Policy (argmax_a Q)')
    if save:
        plt.savefig("2d-policy.png")
    plt.show()

def plot_rewards(N_episodes, episode_reward_list, save=False):
    # Plot Rewards
    plt.plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
    plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 10), label='Average episode reward')
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.title('Total Reward vs Episodes')
    plt.legend()
    plt.grid(alpha=0.3)
    if save:
        plt.savefig("reward-epsilon-softmax.png")
    plt.show()

def plot_lambdas(values, rewards, confidence_list, save=False):
    xs = np.arange(len(values))
    plt.figure(figsize=(6,4))

    plt.errorbar(
        xs,
        rewards,
        yerr=confidence_list,
        marker='o',
        linewidth=2,
        capsize=5
    )

    plt.xticks(xs, values)
    plt.xlabel(r"Eligibility trace  $\lambda$", fontsize=12)
    #plt.xlabel(r"Initial step size  $\alpha$", fontsize=12)
    plt.ylabel("Average total reward", fontsize=12)
    plt.title("Effect of Eligibility trace on Average Return", fontsize=14)
    #plt.title("Effect of step size on Average Return", fontsize=14)

    CONFIDENCE_PASS = -135
    plt.axhline(CONFIDENCE_PASS, linestyle='--', color='red')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig("rewards-lambdas-conf.png")
    plt.show()
    
def plot_alphas(values, rewards, confidence_list, save=False):
    xs = np.arange(len(values))
    plt.figure(figsize=(6,4))

    plt.errorbar(
        xs,
        rewards,
        yerr=confidence_list,
        marker='o',
        linewidth=2,
        capsize=5
    )

    plt.xticks(xs, values)
    #plt.xlabel(r"Eligibility trace  $\lambda$", fontsize=12)
    plt.xlabel(r"Initial step size  $\alpha$", fontsize=12)
    plt.ylabel("Average total reward", fontsize=12)
    #plt.title("Effect of Eligibility trace on Average Return", fontsize=14)
    plt.title("Effect of step size on Average Return", fontsize=14)

    CONFIDENCE_PASS = -135
    plt.axhline(CONFIDENCE_PASS, linestyle='--', color='red')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig("rewards-alphas-conf.png")
    plt.show()

def train_model(env, N_episodes, k, N, W, eta, alpha, gamma, lambda_, m, v, step_decay_thresh, alpha_decay, exploration_strategy, strategy_params):
    
    episode_reward_list = []  # Used to save episodes reward

    if exploration_strategy == 'epsilon_greedy':
        epsilon = strategy_params['epsilon']
        eps_decay = strategy_params['eps_decay']


    
    # Training process
    action_counts = {}
    total_steps = 0
    for i in range(1, N_episodes + 1):
        # Reset enviroment data
        done = False
        truncated = False
        state = scale_state_variables(env.reset()[0], env.observation_space.low, env.observation_space.high)
        total_episode_reward = 0
        
        Z = np.zeros((k, N))                            # eligibility matrix
        action = np.random.randint(0, k)                # initialize action
        phi_s = np.cos(np.pi * np.dot(eta, state))      # compute phi(s_start)
        Q_s = np.dot(W[action], phi_s)                  # initialize Q

        while not (done or truncated):
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise. Truncated is true if you reach 
            # the maximal number of time steps, False else.
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = scale_state_variables(next_state, env.observation_space.low, env.observation_space.high)
            total_steps += 1

            phi_next = np.cos(np.pi * np.dot(eta, next_state))   # compute phi(s)
            Q_next = np.dot(W, phi_next)    # compute Q(s_t+1, a)
        
            if exploration_strategy == 'softmax':
                next_action = softmax_exploration(Q_next, **strategy_params)
            
            elif exploration_strategy == 'ucb':
                next_action = ucb_exploration(Q_next, next_state, action_counts, total_steps)
                action_counts[(tuple(np.round(state, 1)), action)] = \
                    action_counts.get((tuple(np.round(state, 1)), action), 0) + 1
            
            else:  # epsilon_greedy
                next_action = epsilon_greedy(Q_next, epsilon)


            Z *= gamma * lambda_    # update eligibility matrix
            Z[action, :] += phi_s   # add phi(s) to z_a
            Z = np.clip(Z, -5, 5)   # clip Z to avoid exploding gradient

            # TD error
            delta_t = reward + gamma * Q_next[next_action] - Q_s

            # weight update
            if use_momentum:
                v = m*v + delta_t * (Z * alpha)
                W += v

            elif use_nesterov:
                v = m*v + delta_t * (Z * alpha)
                W += m*v + delta_t * (Z * alpha)

            else:
                W += delta_t * (Z * alpha)

            # Update episode reward
            total_episode_reward += reward

            # update Qs and phi
            Q_s = Q_next[next_action]
            phi_s = phi_next

            # Update state and action for next iteration
            state = next_state
            action = next_action

        # apply step decay only when approaching good policy
        if total_episode_reward > step_decay_thresh:
            alpha *= alpha_decay
        
        if exploration_strategy == 'epsilon_greedy':
            epsilon *= eps_decay

        # Append episode reward
        episode_reward_list.append(total_episode_reward)

    # Close environment
    env.close()
    return W, eta, episode_reward_list

def evaluate_different_alphas(alpha_values, env, N_episodes, gamma, lambda_, m, v, step_decay_thresh, alpha_decay, exploration_strategy, strategy_params):
    # initialize eta
    eta = initialize_eta(p, include_bias)

    N = len(eta)
    k = env.action_space.n      # tells you the number of actions
    confidence_list = []
    reward_list = []

    for alpha_start in alpha_values:

        # initialise weights, phi and alpha
        W = initialize_weights(N, k)
        alpha = initialize_alphas(eta, alpha_start)

        W, eta, _ = train_model(env, N_episodes, k, N, W, eta, alpha, gamma, lambda_, m, v, step_decay_thresh, alpha_decay, exploration_strategy, strategy_params)
       
        data = {'W': W, 'N': eta}
        save_W_eta(data)

        reward_eval, confidence = evaluate()
        reward_list.append(reward_eval)
        confidence_list.append(confidence)

    plot_alphas(alpha_values, reward_list, confidence_list)

def evaluate_different_lambdas(lambda_values, env, N_episodes, gamma, lambda_, m, v, step_decay_thresh, alpha_decay, exploration_strategy, strategy_params):
    # initialize eta
    eta = initialize_eta(p, include_bias)

    N = len(eta)
    k = env.action_space.n      # tells you the number of actions
    confidence_list = []
    reward_list = []

    for lambda_ in lambda_values:

        # initialise weights, phi and alpha
        W = initialize_weights(N, k)
        alpha = initialize_alphas(eta, alpha_start)

        W, eta, _ = train_model(env, N_episodes, k, N, W, eta, alpha, gamma, lambda_, m, v, step_decay_thresh, alpha_decay, exploration_strategy, strategy_params)
        
        data = {'W': W, 'N': eta}
        save_W_eta(data)

        reward_eval, confidence = evaluate()
        reward_list.append(reward_eval)
        confidence_list.append(confidence)
        
    plot_lambdas(lambda_values, reward_list, confidence_list)

    
# Parameters
N_episodes = 100       # Number of episodes to run for training
p = 2       # foiurier order
n = 2       # state dimension
gamma = 1   # discount factor
lambda_ = 0.8 # eligibility trace
epsilon = 0.01   # exploration rate
include_bias = 1 # use eta_0 or not

# initialize eta
eta = initialize_eta(p, include_bias)

N = len(eta)
k = env.action_space.n      # tells you the number of actions

# initialise weights, phi and alpha
alpha_start = 0.01
W = initialize_weights(N, k)
alpha = initialize_alphas(eta, alpha_start)


m = 0.1          # momentum 
use_momentum = True
use_nesterov = False
v = np.zeros((k, N)) # initialize v for sgd modifications

# hyper parameters for training
eps_decay = 0.7
step_decay_thresh = -145
alpha_decay = 0.7

# initialize list for saving variables
total_episode_reward_avg = 0

# initialize parameters for exploration strategy
exploration_strategy = 'epsilon_greedy'
  
if exploration_strategy == 'softmax':
    strategy_params = {
    'temperature': 0.5
    }
    
elif exploration_strategy == 'ucb':
    strategy_params = {
    'bonus_scale': 2.0
    }
    action_counts = {}  # Dictionary to track visits
    total_steps = 0
    
elif exploration_strategy == 'epsilon_greedy':
    strategy_params = {
    'epsilon': epsilon,
    'eps_decay': eps_decay

    }


########################################### Train and evaluate for N_episodes, plot rewards, value function and policy (3d) #############################33


normal_run = True
plot_3d_valueFunc = True
if normal_run:

    # THIS WILL OVERWRITE WEIGHTS.PKL!!!!
    W, eta, episode_reward_list = train_model(env, N_episodes, k, N, W, eta, alpha, gamma, lambda_, m, v, step_decay_thresh, alpha_decay ,exploration_strategy, strategy_params)
    plot_rewards(N_episodes, episode_reward_list)
    data = {'W': W, 'N': eta}
    save_W_eta(data)                        # if you want to save weights.pkl
    reward_eval, confidence = evaluate()
    if plot_3d_valueFunc:
        plot_3d_valueFunc_policy(N_episodes, episode_reward_list)
    

########################################### plot over alphas #############################33



alpha_values = [0.001, 0.01, 0.05, 0.1, 0.3]

plot_over_different_alphas = False
if plot_over_different_alphas:

    # THIS WILL OVERWRITE WEIGHTS.PKL!!!!
    evaluate_different_alphas(alpha_values, env, N_episodes, gamma, lambda_, m, v, step_decay_thresh, alpha_decay, exploration_strategy, strategy_params)




########################################### plot over lambdas #############################33
lambda_values = [0.1, 0.3, 0.5, 0.7, 0.9]

plot_over_different_lambdas = False
if plot_over_different_lambdas:
    
    # THIS WILL OVERWRITE WEIGHTS.PKL!!!!
    evaluate_different_lambdas(lambda_values, env, N_episodes, gamma, lambda_, m, v, step_decay_thresh, alpha_decay, exploration_strategy, strategy_params)
