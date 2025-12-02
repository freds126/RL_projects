# Close environment# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

# Load packages
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 100       # Number of episodes to run for training
discount_factor = 1.    # Value of gamma


# Reward
episode_reward_list = []  # Used to save episodes reward


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

"""

initialize eta:
    9x eta_i = (n_1, n_2) (9x2) matrix?
    
    phi(s): 9x1 => phi_i =  cos(pi*eta_i* s)

    weights: w_a is 9x1, a = 3 (?) so W = 3x9 -- this is what we are learning

    Q(s, a) = w^T*phi(s)

    Eligibilitiy traces:
        z_a = 9x1, Z = 9x3 
                                clip between -5, 5!!
        eligibility update:
            if a == a_t:
                z_a = gamma * lambda * z_a + phi(s_t)

            else: 
                z_a = gamma * lambda * z_a 
    TD Error:
       delta_t = r_t + gamma*Q(s_t+1, a_t+1) - Q(s_t, a_t)



    eligibility update:
        if a == a_t:
            z_a = gamma * lambda * z_a + phi(s_t)

        else: 
            z_a = gamma * lambda * z_a 

    weight update:
        W = W + step * delta_t * Z

    step:

"""

best_sofar = {'lambda': 0.8, 
              'epsilon': 0.3, 
              'alpha_start': 0.01, 
              'm': 0.1, 
              'eps_decay': 0.7, 
              'step_decay_thresh': -150, 
              'sgd': True, ''
              'sgd_nesterov': False
              }                             # -111 +- 1.7

best_sofar = {'lambda': 0.8, 
              'epsilon': 0.3, 
              'alpha_start': 0.01, 
              'm': 0.1, 
              'eps_decay': 0.7, 
              'step_decay_thresh': -140, 
              'sgd': True, ''
              'sgd_nesterov': False
              }                             # -114 +- 1.7

### BEST ONE, least variance
best_sofar_113 = {'lambda': 0.8, 
              'epsilon': 0.3, 
              'alpha_start': 0.01, 
              'm': 0.1, 
              'eps_decay': 0.7, 
              'step_decay_thresh': -145, 
              'sgd': True, ''
              'sgd_nesterov': False
              }                             # -113 +- 0.3
import pickle
save = False
with open("best_params.pkl", "wb") as f:
    if save:
        pickle.dump(best_sofar_113, f)


# initialize
p = 2       # foiurier order
n = 2       # state dimension
gamma = 1   # discount factor
lambda_ = 0.8 # eligibility trace
epsilon = 0.3   # exploration rate

eta = np.array([[i, j] for i in range(p+1) for j in range(p+1)])

eta = eta[1:, :]

N = len(eta)

W = np.zeros((k, N)) 

phi = np.ones((N,))

alpha_start = 0.01
alpha = np.ones((N,)) * alpha_start

L2_norm = np.linalg.norm(eta, axis=1)
alpha = alpha/(L2_norm + (L2_norm == 0))    # alpha_i = alpha_i-1/||eta_i||, if ||eta_i|| = 0, alhpa_i = alpha_i-1# step size vector
sgd = True
sgd_nesterov = False
v = np.zeros((k, N))
m = 0.1

eps_decay = 0.7
step_decay_thresh = -145

Train = True
if Train:
    # Training process
    for i in range(N_episodes):
        # Reset enviroment data
        done = False
        truncated = False
        state = scale_state_variables(env.reset()[0])
        total_episode_reward = 0
        Z = np.zeros((k, N)) # eligibility matrix

        action = np.random.randint(0, k)    # initialize action
        phi_s = np.cos(np.pi * np.dot(eta, state))   # compute phi(s)# initialize phi
        Q_s = np.dot(W[action], phi_s) # initialize Q

        while not (done or truncated):
            # Take a random action
            # env.action_space.n tells you the number of actions
            # available
            #action = np.random.randint(0, k)
            
                
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise. Truncated is true if you reach 
            # the maximal number of time steps, False else.
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = scale_state_variables(next_state)

            phi_next = np.cos(np.pi * np.dot(eta, next_state))   # compute phi(s)
            Q_next = np.dot(W, phi_next)    # compute Q(s_t+1, a)

            if np.random.random() < epsilon:            # choose next action epsilon greedily
                next_action = np.random.randint(0, k)
            else:
                next_action = np.argmax(Q_next)              # action leading to maximum Q value

            Z *= gamma * lambda_ # update eligibility matrix
            Z[action, :] += phi_s    # add phi(s) to z_a
            Z = np.clip(Z, -5, 5)   # clip Z to avoid exploding gradient

            # TD error
            delta_t = reward + gamma * Q_next[next_action] - Q_s
            #print(f"iter:  {i}  max: {reward}")

            # weight update
            if sgd:
                v = m*v + delta_t * (Z * alpha)
                W += v

            elif sgd_nesterov:
                v = m*v + delta_t * (Z * alpha)
                W += m*v + delta_t * (Z * alpha)

            else:
                W += delta_t * (Z * alpha)
            #print(f"iter:  {i}  max: {np.max(W)}")
            # Update episode reward
            total_episode_reward += reward
                
            # Update state for next iteration
            state = next_state
            
            Q_s = Q_next[next_action]
            phi_s = phi_next
            action = next_action

        # epsilon decay
        epsilon =  1/((i + 1) **(eps_decay))
        
        if total_episode_reward > step_decay_thresh:
            alpha *= 0.7

        # Append episode reward
        episode_reward_list.append(total_episode_reward)

        # Close environment
        env.close()



plot = True
if plot:

    # Plot Rewards
    plt.plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
    plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 10), label='Average episode reward')
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.title('Total Reward vs Episodes')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    def phi(s):
        # s is unscaled (env state)
        s_scaled = scale_state_variables(s)
        return np.cos(np.pi * (eta @ s_scaled))  # shape (9,)

    def Q_values(s):
        # returns Q(s,a) for all actions as a vector of length 3
        phi_s = phi(s)
        return W @ phi_s  # shape (3,)

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
            Qs = Q_values(s)
            V_values[i, j] = np.max(Qs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(P, V, V_values, linewidth=0, antialiased=True)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('V(s) = max_a Q(s,a)')
    ax.set_title('Approximate Value Function of Learned Policy')
    plt.show()

    policy = np.zeros((n_vel, n_pos))

    for i in range(n_vel):
        for j in range(n_pos):
            s = np.array([P[i,j], V[i,j]])
            Qs = Q_values(s)
            policy[i,j] = np.argmax(Qs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(P, V, policy, linewidth=0, antialiased=True)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Action (0=left,1=none,2=right)')
    ax.set_title('Optimal Policy over State Space')
    plt.show()

    plt.figure()
    plt.pcolormesh(P, V, policy, shading='auto')
    cbar = plt.colorbar()
    cbar.set_label('Action (0=left,1=none,2=right)')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Optimal Policy (argmax_a Q)')
    plt.show()





import pickle

data = {'W': W, 'N': eta}

save = True
with open("weights.pkl", "wb") as f:
    if save:
        pickle.dump(data, f)
        