# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.

# Load packages
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import EpsilonGreedyAgent, RandomAgent
import warnings
from collections import deque, namedtuple
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
from QNetwork import QNetwork, DuelingQNetwork

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state',
'done'])

class ExperienceReplayBuffer:
    
    def __init__(self, buffer_size: int):
        self.buffer = deque(maxlen=buffer_size)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, n: int, latest_exp: Experience = None):
        batch = random.sample(self.buffer, n) 
        if latest_exp:
            batch.append(latest_exp)
        
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )
    

### HELPERS ###


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def epsilon_linear(k, Z, epsilon_max=0.99, epsilon_min=0.05):
    """
    Linear epsilon decay.

    Args:
        k (int): current episode (1-indexed)
        Z (int): number of decay episodes (typically 0.9–0.95 * total episodes)
        epsilon_max (float): initial epsilon
        epsilon_min (float): minimum epsilon

    Returns:
        float: epsilon_k
    """
    if Z <= 1:
        return epsilon_min

    epsilon = epsilon_max - (epsilon_max - epsilon_min) * (k - 1) / (Z - 1)
    return max(epsilon_min, epsilon)

def epsilon_exponential(k, Z, epsilon_max=0.99, epsilon_min=0.05):
    """
    Exponential epsilon decay.

    Args:
        k (int): current episode (1-indexed)
        Z (int): number of decay episodes (typically 0.9–0.95 * total episodes)
        epsilon_max (float): initial epsilon
        epsilon_min (float): minimum epsilon

    Returns:
        float: epsilon_k
    """
    if Z <= 1:
        return epsilon_min

    decay_rate = (epsilon_min / epsilon_max) ** ((k - 1) / (Z - 1))
    epsilon = epsilon_max * decay_rate
    return max(epsilon_min, epsilon)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def save_model(q_network, filename="neural-network-1.pth"):
    q_network.eval() 
    # Move to CPU
    q_network_cpu = q_network.to("cpu")
    torch.save(q_network_cpu, filename)#, _use_new_zipfile_serialization=False)  
    print("Model saved successfully!")

def load_model(path="neural-network-1.pth"):
    """
    Loads a trained QNetwork
    """
    model = torch.load(path, weights_only=False)
    model.eval()
    return model

def save_params(params):
    torch.save(params, "training_params.pth")
    print("Saved training_params.pth")


### Plots and such ###


def plot_avg_rewards_n_steps(episode_reward_list, episode_number_of_steps, n, filename, save = False):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))

    ax[0].plot(range(1, len(episode_reward_list) + 1), episode_reward_list, label="Episode reward")
    ax[0].plot(range(1, len(episode_reward_list) + 1),
               running_average(episode_reward_list, n),
               label="Avg. episode reward")
    ax[0].set_xlabel("Episodes")
    ax[0].set_ylabel("Total reward")
    ax[0].set_title("Total Reward vs Episodes")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(range(1, len(episode_number_of_steps) + 1), episode_number_of_steps, label="Steps per episode")
    ax[1].plot(range(1, len(episode_number_of_steps) + 1),
               running_average(episode_number_of_steps, n),
               label="Avg. number of steps per episode")
    ax[1].set_xlabel("Episodes")
    ax[1].set_ylabel("Total number of steps")
    ax[1].set_title("Total number of steps vs Episodes")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    if save:
        plt.savefig(filename)
    plt.show()

def plot_reward_sweep(runs, n_running_avg=50, title="", size=20, save=False):
    """
    runs: list of dicts, each like:
      {"label": "N=200", "rewards": [...]} 
    """
    plt.figure(figsize=(10, 6))
    for run in runs:
        rewards = run["rewards"]
        plt.plot(running_average(rewards, n_running_avg), label=run["label"], linewidth=2)
    plt.xlabel("Episode", fontsize=size)
    plt.ylabel(f"Reward (running avg, N={n_running_avg})", fontsize=size)
    plt.title(title, fontsize=size)
    plt.legend(fontsize=size)
    plt.grid(alpha=0.3)
    plt.tight_layout(pad=2)
    if save:
        plt.savefig(f"{title}.png")
    plt.show()

def plot3d_grid(model, size=20, save=False):
    from mpl_toolkits.mplot3d import Axes3D

    Ny, Nw = 100, 150

    y_vals = np.linspace(0.0, 2.5, Ny)
    w_vals = np.linspace(-2*np.pi, 2*np.pi, Nw)

    Y, W = np.meshgrid(y_vals, w_vals, indexing = "ij")

    states = np.zeros((Ny*Nw, 8), dtype=np.float32)
    states[:, 1] = Y.reshape(-1)
    states[:, 4] = W.reshape(-1)

    states_t = torch.from_numpy(states)

    with torch.no_grad():
        Q = model(states_t)
        V = Q.max(dim=1).values.numpy()
        A = Q.argmax(dim=1).numpy()

    V_grid = V.reshape(Ny, Nw)
    A_grid = A.reshape(Ny, Nw)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(Y, W, V_grid, linewidth=0, antialiased=True)

    ax.set_xlabel("y (height)", fontsize=size)
    ax.set_ylabel(r"$\theta$ (angle)  [rad]", fontsize=size)
    ax.set_zlabel(r"$\max_a$ Q(s,a)", fontsize=size)
    ax.set_title(r"Value surface: $\max_a$ $Q_{\theta}(s(y,\theta), a)$", fontsize=size)
    
    if save:
        plt.savefig("V_3d_plot.png")

    plt.show()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(Y, W, A_grid.astype(float), linewidth=0, antialiased=False)

    ax.set_xlabel("y (height)", fontsize=size)
    ax.set_ylabel(r"$\theta$ (angle)  [rad]", fontsize=size)
    ax.set_zlabel(r"arg$\max_a Q(s,a)$", fontsize=size)
    ax.set_title(r"Greedy action: arg$\max_a$ $Q_{\theta} (s(y,\theta), a)$", fontsize=size)

    if save:
        plt.savefig("Q_3d_plot.png")

    plt.show()


### run different tests ###


def run_discount_factor_comparison(params, discount_factors, save):

    for discount_factor in discount_factors:
        params["discount_factor"] = discount_factor
        filename = f"new_avg_reward_with_discount_{discount_factor}.png"

        q_network, epsiode_reward_list, epsiode_step_list = train_dqn(params)
        plot_avg_rewards_n_steps(epsiode_reward_list, epsiode_step_list, params["n_ep_running_average"], filename, save)

def run_N_episodes_comparison(params, episode_counts, size=20, save=False):

    base_params = params.copy()

    runs = []
    for N in episode_counts:
        run_params = base_params.copy()
        run_params["N_episodes"] = N

        qnet, rewards, steps = train_dqn(run_params)

        runs.append({
            "label": f"N={N}",
            "rewards": rewards
        })

    if save:
        torch.save(runs, "N_ep_runs.pth")

    plot_reward_sweep(
        runs,
        base_params["n_ep_running_average"],
        f"Effect of number of training episodes)", 
        size,
        save
    )

def run_buffersize_comparison(params, buffer_counts, size=20, save=False):

    base_params = params.copy()

    runs = []
    for buff in buffer_counts:
        run_params = base_params.copy()
        run_params["BUFFER_SIZE"] = buff

        qnet, rewards, steps = train_dqn(run_params)

        runs.append({
            "label": f"BUFFER_SIZE={buff}",
            "rewards": rewards
        })

    if save:
        torch.save(runs, "buffersize_runs.pth")

    plot_reward_sweep(
        runs,
        base_params["n_ep_running_average"],
        f"Effect of buffer size", 
        size, 
        save
    )

def check_solution(env, agent, N_EPISODES = 50, epsilon = 0.0):
    # Import and initialize Mountain Car Environment
    # If you want to render the environment while training run instead:
    #env = gym.make('LunarLander-v3', render_mode = "human")

    env.reset()

    # Parameters
 
    CONFIDENCE_PASS = 50

    # Reward
    episode_reward_list = []  # Used to store episodes reward

    # Simulate episodes
    print('Checking solution...')
    EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
    for i in EPISODES:
        EPISODES.set_description("Episode {}".format(i))
        # Reset enviroment data
        done, truncated = False, False
        state = env.reset()[0]
        total_episode_reward = 0.
        while not (done or truncated):
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            
            # Choose action via the agent interface
            if isinstance(agent, EpsilonGreedyAgent):
                action = agent.forward(state, epsilon=epsilon)
            else:
                action = agent.forward(state)
            
            next_state, reward, done, truncated, _ = env.step(action)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state

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
        print("Your policy did not pass the test! The average reward of your policy needs to be greater than {} with 95% confidence".format(CONFIDENCE_PASS))
    return avg_reward, confidence


### main DQN training loop ###


def train_dqn(params):
    device = get_device()
    print(f"Using device: {device}")

    # Import and initialize the discrete Lunar Lander Environment
    env = gym.make('LunarLander-v3')
    # If you want to render the environment while training run instead:
    # env = gym.make('LunarLander-v3', render_mode = "human")

    env.reset()

    n_actions = env.action_space.n
    dim_state = len(env.observation_space.high)

    epsilon_Z = int(params["epsilon_Z_frac"] * params["N_episodes"])
    epsilon, epsilon_max = params["epsilon_max"], params["epsilon_max"]
    epsilon_min = params["epsilon_min"]
    N_episodes = params["N_episodes"]
    discount_factor = params["discount_factor"]
    
    train_every = params["train_every"]
    batch_size = params["batch_size"]
    min_buf = params["min_buffer_batches"] * batch_size
    target_update_freq = params["target_update_freq"]
    grad_clip = params["grad_clip"]
    
    n_ep_running_average = params["n_ep_running_average"]
    stop_avg_reward = params["stop_avg_reward"]

    use_cer = params["use_CER"]
 
    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode

    # intialize buffer
    buffer = ExperienceReplayBuffer(params["BUFFER_SIZE"])

    # initialize Q-networks
    q_network = QNetwork(dim_state, n_actions, params["latent_dim"]).to(device)
    target_network = QNetwork(dim_state, n_actions, params["latent_dim"]).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    # Epsilon greedy agent initialization
    agent = EpsilonGreedyAgent(n_actions, q_network, device)

    # initalize optimizer
    optimizer = optim.Adam(q_network.parameters(), lr=params["lr"])

    ### Training process

    # trange is an alternative to range in python, from the tqdm library
    # It shows a nice progression bar that you can update with useful information
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    global_t = 1
    nr_target_updates = 0

    for i in EPISODES:
        # Reset enviroment data and initialize variables
        done, truncated = False, False
        state = env.reset()[0]
        total_episode_reward = 0.0
        t = 0

        while not (done or truncated):
            # Take a epsilon greedy action
            action = agent.forward(state, epsilon)

            # Get next state and reward
            next_state, reward, done, truncated, _ = env.step(action)

            # append to buffer
            buffer.append(Experience(state, action, reward, next_state, done))

            if (global_t % train_every == 0 and len(buffer) >= min_buf):        # if buffer has filled up enough
                
                if use_cer:
                    # Combined Experience Replay
                    latest_exp = Experience(state, action, reward, next_state, done)
                    
                    # Sample from replay buffer
                    states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size, latest_exp)
                else:
                    # Sample from replay buffer
                    states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size)
                
                # make tensors, and unsqueeze
                states      = torch.from_numpy(states).float().to(device)
                next_states = torch.from_numpy(next_states).float().to(device)
                actions = torch.from_numpy(actions).long().to(device).unsqueeze(1)
                rewards = torch.from_numpy(rewards).float().to(device)
                dones   = torch.from_numpy(dones).float().to(device)

                # compute current Q(s,a)
                q_values_curr = q_network(states).gather(1, actions).squeeze(1)

                # compute targets - max(Q(s', a'))
                with torch.no_grad():
                    max_next_q_values = target_network(next_states).amax(dim=1)
                    targets = rewards + discount_factor * (max_next_q_values) * (1 - dones)

                # mse loss
                loss = nn.functional.mse_loss(q_values_curr, targets)
                
                # taking graident step
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to avoid exploding gradients
                nn.utils.clip_grad_norm_(q_network.parameters(), grad_clip) 
                optimizer.step()
                
                # update target network
                if global_t % target_update_freq == 0:
                    target_network.load_state_dict(q_network.state_dict())
                    nr_target_updates += 1

            # Update episode reward
            total_episode_reward += reward
            
            # Update state for next iteration
            state = next_state
            t+= 1
            global_t += 1

        # epsilon decay
        if params["use_exp_eps_decay"]:
            epsilon = epsilon_exponential(i+1, epsilon_Z, epsilon_max, epsilon_min)
        else:
            epsilon = epsilon_linear(i+1, epsilon_Z, epsilon_max, epsilon_min)
        

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        avg_reward = running_average(episode_reward_list, n_ep_running_average)[-1]
        avg_steps = running_average(episode_number_of_steps, n_ep_running_average)[-1]

        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            avg_reward, avg_steps
            ))
        
        # early stoppage if good policy is learned
        if avg_reward > stop_avg_reward:
            print(f"Agent successfully trained - Avg Reward: {avg_reward} after {global_t} iterations!")
            break

    print(f"Nr of target updates:    {nr_target_updates}")

    # Close environment
    env.close()

    return q_network, episode_reward_list, episode_number_of_steps

def train_dueling_dqn(params):
    device = get_device()
    print(f"Using device: {device}")

    # Import and initialize the discrete Lunar Lander Environment
    env = gym.make('LunarLander-v3')
    # If you want to render the environment while training run instead:
    # env = gym.make('LunarLander-v3', render_mode = "human")

    env.reset()

    n_actions = env.action_space.n
    dim_state = len(env.observation_space.high)

    epsilon_Z = int(params["epsilon_Z_frac"] * params["N_episodes"])
    epsilon, epsilon_max = params["epsilon_max"], params["epsilon_max"]
    epsilon_min = params["epsilon_min"]
    N_episodes = params["N_episodes"]
    discount_factor = params["discount_factor"]
    
    train_every = params["train_every"]
    batch_size = params["batch_size"]
    min_buf = params["min_buffer_batches"] * batch_size
    target_update_freq = params["target_update_freq"]
    grad_clip = params["grad_clip"]
    
    n_ep_running_average = params["n_ep_running_average"]
    stop_avg_reward = params["stop_avg_reward"]

    use_cer = params["use_CER"]
 
    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode

    # intialize buffer
    buffer = ExperienceReplayBuffer(params["BUFFER_SIZE"])

    # initialize Q-networks
    q_network = DuelingQNetwork(dim_state, n_actions, params["latent_dim"]).to(device)
    target_network = DuelingQNetwork(dim_state, n_actions, params["latent_dim"]).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    # Epsilon greedy agent initialization
    agent = EpsilonGreedyAgent(n_actions, q_network, device)

    # initalize optimizer
    optimizer = optim.Adam(q_network.parameters(), lr=params["lr"])

    ### Training process

    # trange is an alternative to range in python, from the tqdm library
    # It shows a nice progression bar that you can update with useful information
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    global_t = 1
    nr_target_updates = 0

    for i in EPISODES:
        # Reset enviroment data and initialize variables
        done, truncated = False, False
        state = env.reset()[0]
        total_episode_reward = 0.0
        t = 0

        while not (done or truncated):
            # Take a random action
            action = agent.forward(state, epsilon)

            # Get next state and reward
            next_state, reward, done, truncated, _ = env.step(action)

            # append to buffer
            buffer.append(Experience(state, action, reward, next_state, done))

            if (global_t % train_every == 0 and len(buffer) >= min_buf):        # if buffer has filled up enough
                
                if use_cer:
                    # Combined Experience Replay
                    latest_exp = Experience(state, action, reward, next_state, done)
                    
                    # Sample from replay buffer
                    states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size, latest_exp)
                else:
                    # Sample from replay buffer
                    states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size)
                
                # make tensors, and unsqueeze
                states      = torch.from_numpy(states).float().to(device)
                next_states = torch.from_numpy(next_states).float().to(device)
                actions = torch.from_numpy(actions).long().to(device).unsqueeze(1)
                rewards = torch.from_numpy(rewards).float().to(device)
                dones   = torch.from_numpy(dones).float().to(device)

                # compute current Q(s,a)
                q_values_curr = q_network(states).gather(1, actions).squeeze(1)

                # compute targets - max(Q(s', a'))
                with torch.no_grad():
                    max_next_q_values = target_network(next_states).amax(dim=1)
                    targets = rewards + discount_factor * (max_next_q_values) * (1 - dones)

                # mse loss
                loss = nn.functional.mse_loss(q_values_curr, targets)
                
                # taking graident step
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to avoid exploding gradients
                nn.utils.clip_grad_norm_(q_network.parameters(), grad_clip) 
                optimizer.step()
                
                # update target network
                if global_t % target_update_freq == 0:
                    target_network.load_state_dict(q_network.state_dict())
                    nr_target_updates += 1

            # Update episode reward
            total_episode_reward += reward
            
            # Update state for next iteration
            state = next_state
            t+= 1
            global_t += 1

        # epsilon decay
        if params["use_exp_eps_decay"]:
            epsilon = epsilon_exponential(i+1, epsilon_Z, epsilon_max, epsilon_min)
        else:
            epsilon = epsilon_linear(i+1, epsilon_Z, epsilon_max, epsilon_min)
        

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        avg_reward = running_average(episode_reward_list, n_ep_running_average)[-1]
        avg_steps = running_average(episode_number_of_steps, n_ep_running_average)[-1]

        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            avg_reward, avg_steps
            ))
        
        # early stoppage if good policy is learned
        if avg_reward > stop_avg_reward:
            print(f"Agent successfully trained - Avg Reward: {avg_reward} after {global_t} iterations!")
            break

    print(f"Nr of target updates:    {nr_target_updates}")

    # Close environment
    env.close()

    return q_network, episode_reward_list, episode_number_of_steps

if __name__ == '__main__':
    """
    You have to use torch.load('neural-network-1.pth', weights_only=False) to make the check solution script work!!!
    """
    params = {
    
        "N_episodes": 500,          # Nr of epsiodes to train for
        "discount_factor": 0.99,    
        "n_ep_running_average": 50, # running average size


        "BUFFER_SIZE": 30000,       # experience buffer size
        "batch_size": 64,           # batch size

        "epsilon_max": 0.99,        # starting value/ max value of epsilon
        "epsilon_min": 0.01,        # min value of epsilon
        "epsilon_Z_frac": 0.95,     # Z of epsilon decay - number of epsiodes over which epsilon decays
        "use_exp_eps_decay": 0,      # use exponential decay - default linear
        
        "target_update_freq": 500,  # how often target_network = q_network
        "train_every": 4,           # train q_network every 4 iterations  
        "min_buffer_batches": 100,  # fill experienceBuffer with min_buffer_batches * batch_size

        "lr": 5e-4,                 # learning rate
        "latent_dim": 128,          # latent dimension of nn
        "grad_clip": 2.0,           # max_norm of gradient

        "stop_avg_reward": 300,     # early stop threshold
        "use_CER": True,            # use combine replay experience
    }
    
    run_discount_comp = False
    if run_discount_comp:
        discount_factors = [0.99, 1, 0.5]
        run_discount_factor_comparison(params, discount_factors, save=False)
    
    run_episode_comp = False
    if run_episode_comp:
        episode_counts = [200, 500, 800]  
        run_N_episodes_comparison(params, episode_counts, save=False) # this plots as well from inside the function :)

        #runs = torch.load("N_ep_runs.pth", weights_only=False)
        #plot_reward_sweep(runs, 50, "Effect of number of training episodes", size=20, save=False)

    run_buffer_comp = False
    if run_buffer_comp:        
        buffer_counts = [2000, 15000, 30000]
        run_buffersize_comparison(params, buffer_counts, size=20, save=False)

    plot3d = False
    if plot3d:
        model = load_model()
        model.eval()
        plot3d_grid(model, size=20, save=False)

    compare_agents = False
    if compare_agents:
        device = get_device()
        env = gym.make('LunarLander-v3')
        n_actions = env.action_space.n

        model = load_model().to(device)
        model.eval()
        
        randomAgent = RandomAgent(n_actions)        
        greedyAgent = EpsilonGreedyAgent(n_actions, model, device)

        print("-------------------RANDOM AGENT -----------------------\n\n")
        check_solution(env, randomAgent)

        print("\n------------------- GREEDY AGENT -----------------------\n\n")
        check_solution(env, greedyAgent)

    savePath = "neural-network-1.pth"
    normal_run = True
    if normal_run:
        qnet, rewards, steps = train_dqn(params)
        plot_avg_rewards_n_steps(rewards, steps, 50, filename="", save=False)
        save_model(qnet, savePath)

    dueling_run = False
    if dueling_run:
        qnet, rewards, steps = train_dueling_dqn(params)
        plot_avg_rewards_n_steps(rewards, steps, 50, filename="", save=False)
        save_model(qnet, savePath)
    
    #save_params(params)
    