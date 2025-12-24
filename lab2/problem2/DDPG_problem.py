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
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import RandomAgent, DDPGAgent, LowPassNoise, UncorrNormalNoise
from DDPGNetworks import Actor, Critic
from DDPG_soft_updates import soft_updates
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state',
'done'])

class ExperienceReplayBuffer:
    
    def __init__(self, buffer_size: int):
        self.buffer = deque(maxlen=buffer_size)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, n: int):
        batch = random.sample(self.buffer, n)    
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )
    
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

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_requires_grad(net, flag):
    for p in net.parameters():
        p.requires_grad = flag

def save_model(network, filename="neural-network-2.pth"):
    network.eval() 
    # Move to CPU
    network_cpu = network.to("cpu")
    torch.save(network_cpu, filename)#, _use_new_zipfile_serialization=False)  
    print("Model saved successfully!")

def load_model(path="neural-network-1.pth"):
    """
    Loads a trained QNetwork
    """
    model = torch.load(path, weights_only=False)
    model.eval()
    return model

def save(params, filename="training_params.pth"):
    torch.save(params, filename)
    print("Saved training_params.pth")

def load(filename):
    return torch.load(filename, weights_only=False)

def plot_rewards_steps(episode_reward_list, episode_number_of_steps, filename="rewards_steps_epsiodes.png", save=False, size=20, n_ep_running_average=50):
    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, len(episode_reward_list) + 1)], episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, len(episode_reward_list) + 1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes', fontsize=size)
    ax[0].set_ylabel('Total reward', fontsize=size)
    ax[0].set_title('Total Reward vs Episodes', fontsize=size)
    ax[0].legend(fontsize=size)
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, len(episode_number_of_steps) + 1)], episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, len(episode_number_of_steps)+1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes', fontsize=size)
    ax[1].set_ylabel('Total number of steps', fontsize=size)
    ax[1].set_title('Total number of steps vs Episodes', fontsize=size)
    ax[1].legend(fontsize=size)
    ax[1].grid(alpha=0.3)
    plt.tight_layout(pad=2)

    if save:
        plt.savefig(filename)
    plt.close(fig)
    #plt.show()

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
    plt.close()

    #plt.show()

def plot3d_grid(actor, critic, size=20, save=False):
    
    Ny, Nw = 100, 150
    y_vals = np.linspace(0.0, 1.5, Ny)
    w_vals = np.linspace(-np.pi, np.pi, Nw)

    Y, W = np.meshgrid(y_vals, w_vals, indexing="ij")

    states = np.zeros((Ny * Nw, 8), dtype=np.float32)
    states[:, 1] = Y.reshape(-1)
    states[:, 4] = W.reshape(-1)

    states_t = torch.from_numpy(states)

    with torch.no_grad():
        A = actor(states_t)                  # (N, act_dim)
        V = critic(states_t, A)              # (N, 1) or (N,)

    V_grid = V.squeeze(-1).reshape(Ny, Nw)
    A_grid = A[:, 0].reshape(Ny, Nw)          # choose action dim

    # ---- VALUE PLOT ----
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Y, W, V_grid, linewidth=0, antialiased=True)

    ax.set_xlabel("y (height)", fontsize=size)
    ax.set_ylabel(r"$\theta$ (angle)  [rad]", fontsize=size)
    ax.set_zlabel(r"$Q(s,\pi(s))$", fontsize=size)
    ax.set_title(r"Value surface: $Q(s(y,\theta), \pi(s))$", fontsize=size)

    if save:
        plt.savefig("V_3d_plot-2.png")
    plt.show()

    # ---- ACTION PLOT ----
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Y, W, A_grid, linewidth=0, antialiased=True)

    ax.set_xlabel("y (height)", fontsize=size)
    ax.set_ylabel(r"$\theta$ (angle)  [rad]", fontsize=size)
    ax.set_zlabel(r"$\pi_0(s)$", fontsize=size)
    ax.set_title(r"Actor action surface: $\pi_0(s(y,\theta))$", fontsize=size)

    if save:
        plt.savefig("action_3d_plot-2.png")
    plt.show()

def check_solution(env, agent, N_EPISODES = 50):
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
            if isinstance(agent, DDPGAgent):
                action = agent.forward(state, explore=False)
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


def fill_buffer(env, buffer, N):
    t = 0
    while(t < N):
        # Reset enviroment data
        done, truncated = False, False
        state = env.reset()[0]
        while not (done or truncated):
            # Take a random action
            action = env.action_space.sample()

            # Get next state and reward
            next_state, reward, done, truncated, _ = env.step(action)

            # append to buffer
            buffer.append(Experience(state, action, reward, next_state, done))
            t += 1
    print(f"Filled up buffer with {len(buffer)} experiences!")

def train_ddpg(params):
    # Import and initialize Mountain Car Environment
    env = gym.make('LunarLanderContinuous-v3')
    # If you want to render the environment while training run instead:
    # env = gym.make('LunarLanderContinuous-v3', render_mode = "human")

    env.reset()

    # Parameters
    N_episodes = params["N_episodes"]            # Number of episodes to run for training
    discount_factor = params["discount_factor"]         # Value of gamma
    n_ep_running_average = params["n_ep_running_average"]      # Running average of 50 episodes
    m = len(env.action_space.high) # dimensionality of the action
    dim_state = len(env.observation_space.high)

    buffer_size = params["buffer_size"]
    batch_size = params["batch_size"]
    grad_clip = params["grad_clip"] 
    update_actor_every = params["update_actor_every"]
    tau = params["tau"]

    # parameters for noise
    sigma = params["sigma_noise"]
    mu = params["mu_noise"]

    device = get_device()

    # Reward
    episode_reward_list = []  # Used to save episodes reward
    episode_number_of_steps = []

    # actor networks initialization
    actor_net = Actor(dim_state, m).to(device)
    #actor_net.apply(init_actor_weights)
    actor_target = Actor(dim_state, m).to(device)
    actor_target.load_state_dict(actor_net.state_dict())
    actor_target.eval()

    # critc networks initialization
    critic_net = Critic(dim_state, m).to(device)
    #critic_net.apply(init_critic_weights)
    critic_target = Critic(dim_state, m).to(device)
    critic_target.load_state_dict(critic_net.state_dict())
    critic_target.eval()

    optimizer_critic = optim.Adam(critic_net.parameters(), lr=params["lr_critic"])
    optimizer_actor = optim.Adam(actor_net.parameters(), lr=params["lr_actor"])

    # initalize noise
    noise = LowPassNoise(m, device, sigma, mu)

    # DDPG Agent initialization
    agent = DDPGAgent(m, actor_net, device, noise)

    # initialize buffer
    buffer = ExperienceReplayBuffer(buffer_size)

    # Training process
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    global_t = 1
    avg_reward = 0
    update_count = 0
    start_steps = params["start_steps"]
    start_update = params["start_update"]

    for i in EPISODES:
        # Reset enviroment data
        done, truncated = False, False
        state = env.reset()[0]
        total_episode_reward = 0.
        t = 0
        noise.reset()
        while not (done or truncated):
            # Take an action with added noise
            if global_t > start_steps:
                action = agent.forward(state)
            else:
                action = env.action_space.sample() 

            # Get next state and reward
            next_state, reward, done, truncated, _ = env.step(action)

            # append to buffer
            buffer.append(Experience(state, action, reward, next_state, done))
            
            if (global_t > start_update and len(buffer) >= batch_size):
                states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size)
                
                # make tensors, and unsqueeze
                states      = torch.from_numpy(states).float().to(device)
                next_states = torch.from_numpy(next_states).float().to(device)
                actions = torch.from_numpy(actions).float().to(device)
                rewards = torch.from_numpy(rewards).float().to(device)
                dones   = torch.from_numpy(dones).float().to(device)           

                # compute current q_vals from critic
                q_vals_curr = critic_net(states, actions).squeeze() # (B,1)
                
                # compute targets from target net
                with torch.no_grad():
                    next_actions = actor_target(next_states)
                    next_q_vals = critic_target(next_states, next_actions).squeeze()
                    targets = rewards + (1 - dones) * next_q_vals * discount_factor
                
                # do backpropogation on critic
                critic_loss = nn.functional.mse_loss(targets, q_vals_curr)
                optimizer_critic.zero_grad()
                critic_loss.backward()

                # Clip gradients to avoid exploding gradients
                nn.utils.clip_grad_norm_(critic_net.parameters(), max_norm=grad_clip)
                optimizer_critic.step()

                if global_t % update_actor_every == 0:
                    
                    actions_theta = actor_net(states)
                    q_vals_theta = critic_net(states, actions_theta).squeeze()

                    # update actor
                    loss_actor = -q_vals_theta.mean()
                    optimizer_actor.zero_grad()
                    loss_actor.backward()

                    nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=grad_clip)
                    optimizer_actor.step()

                    # update target networks
                    soft_updates(critic_net, critic_target, tau)
                    soft_updates(actor_net, actor_target, tau)
                update_count += 1

            # Update episode reward
            total_episode_reward += reward
            

            # Update state for next iteration
            state = next_state
            t+= 1
            global_t += 1

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        avg_reward = running_average(episode_reward_list, n_ep_running_average)[-1]
        avg_steps = running_average(episode_number_of_steps, n_ep_running_average)[-1]
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{} - update_count: {}".format(
            i, total_episode_reward, t,
            avg_reward,
            avg_steps, update_count))
        if avg_reward > 240:
            print(f"Successfully trained agent to avg reward of {avg_reward} after {global_t} iterations")
            break

    # Close environment
    env.close()
    return actor_net, critic_net, episode_reward_list, episode_number_of_steps


def train_ddpg_chunking(params):
    # Import and initialize Mountain Car Environment
    env = gym.make('LunarLanderContinuous-v3')
    # If you want to render the environment while training run instead:
    # env = gym.make('LunarLanderContinuous-v3', render_mode = "human")

    env.reset()

    # Parameters
    N_episodes = params["N_episodes"]            # Number of episodes to run for training
    discount_factor = params["discount_factor"]         # Value of gamma
    n_ep_running_average = params["n_ep_running_average"]      # Running average of 50 episodes
    m = len(env.action_space.high) # dimensionality of the action
    dim_state = len(env.observation_space.high)

    buffer_size = params["buffer_size"]
    batch_size = params["batch_size"]
    grad_clip = params["grad_clip"] 
    update_actor_every = params["update_actor_every"]
    train_every = params["train_every"]
    min_buf = params["min_buf"]
    tau = params["tau"]

    # parameters for noise
    sigma = params["sigma_noise"]
    mu = params["mu_noise"]


    device = get_device()

    # Reward
    episode_reward_list = []  # Used to save episodes reward
    episode_number_of_steps = []

    # actor networks initialization
    actor_net = Actor(dim_state, m).to(device)
    #actor_net.apply(init_actor_weights)
    actor_target = Actor(dim_state, m).to(device)
    actor_target.load_state_dict(actor_net.state_dict())
    actor_target.eval()

    # critc networks initialization
    critic_net = Critic(dim_state, m).to(device)
    #critic_net.apply(init_critic_weights)
    critic_target = Critic(dim_state, m).to(device)
    critic_target.load_state_dict(critic_net.state_dict())
    critic_target.eval()

    optimizer_critic = optim.Adam(critic_net.parameters(), lr=params["lr_critic"])
    optimizer_actor = optim.Adam(actor_net.parameters(), lr=params["lr_actor"])

    # initalize noise
    #noise = LowPassNoise(m, device, sigma, mu)
    noise = UncorrNormalNoise(m, device, sigma)

    # DDPG Agent initialization
    agent = DDPGAgent(m, actor_net, device, noise)

    # initialize buffer
    buffer = ExperienceReplayBuffer(buffer_size)

    # Training process
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    global_t = 1
    avg_reward = 0
    start_update = 1000
    start_steps = 10000
    update_count = 0

    #fill_buffer(env, buffer, N=min_buf)
    for i in EPISODES:
        # Reset enviroment data
        done, truncated = False, False
        state = env.reset()[0]
        total_episode_reward = 0.
        t = 0
        noise.reset()
        while not (done or truncated):

            # Take an action with added noise
            if global_t > start_steps:
                action = agent.forward(state)
            else:
                action = env.action_space.sample() 
            # Get next state and reward
            next_state, reward, done, truncated, _ = env.step(action)

            # append to buffer
            buffer.append(Experience(state, action, reward, next_state, done))

            if (global_t % train_every == 0 and global_t > start_update):
                for update_idx in range(train_every):
                    states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size)
                    
                    # make tensors, and unsqueeze
                    states      = torch.from_numpy(states).float().to(device)
                    next_states = torch.from_numpy(next_states).float().to(device)
                    actions = torch.from_numpy(actions).float().to(device)
                    rewards = torch.from_numpy(rewards).float().to(device)
                    dones   = torch.from_numpy(dones).float().to(device)         

                    # CRITIC UPDATE
                    # compute current q_vals from critic
                    q_vals_curr = critic_net(states, actions).squeeze() # (B,1)
                    
                    # compute targets from target net
                    with torch.no_grad():
                        next_actions = actor_target(next_states)
                        next_q_vals = critic_target(next_states, next_actions).squeeze()
                        targets = rewards + (1 - dones) * discount_factor * next_q_vals 
                    
                    # do backpropogation on critic
                    critic_loss = nn.functional.mse_loss(q_vals_curr, targets)
                    optimizer_critic.zero_grad()
                    critic_loss.backward()

                    # Clip gradients to avoid exploding gradients
                    nn.utils.clip_grad_norm_(critic_net.parameters(), max_norm=grad_clip)
                    optimizer_critic.step()
                    update_count += 1
                    # ACTOR UPDATE
                    if update_idx % (update_actor_every) == 0:

                        set_requires_grad(critic_net, False)

                        actions_theta = actor_net(states)
                        q_vals_theta = critic_net(states, actions_theta).squeeze()

                        loss_actor = -q_vals_theta.mean()
                        optimizer_actor.zero_grad()
                        loss_actor.backward()

                        nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=grad_clip)
                        optimizer_actor.step()

                        set_requires_grad(critic_net, True)

                        # update target networks
                        soft_updates(critic_net, critic_target, tau)
                        soft_updates(actor_net, actor_target, tau)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t+= 1
            global_t += 1

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        avg_reward = running_average(episode_reward_list, n_ep_running_average)[-1]
        avg_steps = running_average(episode_number_of_steps, n_ep_running_average)[-1]
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{} - update_count: {}".format(
            i, total_episode_reward, t,
            avg_reward,
            avg_steps, update_count))
        if avg_reward > 240:
            print(f"Successfully trained agent to avg reward of {avg_reward} after {global_t} iterations")
            break

    # Close environment
    env.close()
    return actor_net, critic_net, episode_reward_list, episode_number_of_steps

def run_discount_factor_comparison(params, discount_factors, save):
    torch.manual_seed(100)
    np.random.seed(100)
    random.seed(100)
    base_params = params.copy()
    runs = []
    for discount_factor in discount_factors:
        run_params = base_params.copy()

        run_params["discount_factor"] = discount_factor
        filename = f"new_avg_reward_with_discount_{discount_factor}.png"

        actor, critic, epsiode_reward_list, epsiode_step_list = train_ddpg(run_params)
        plot_rewards_steps(epsiode_reward_list, epsiode_step_list, filename, save)

        runs.append({
            "label": f"gamma={discount_factor}",
            "rewards": epsiode_reward_list
        })

        del actor, critic
        torch.cuda.empty_cache()
    if save:
        torch.save(runs, "discount_factors_runs.pth")

def run_N_episodes_comparison(params, episode_counts, size=20, save=False):
    torch.manual_seed(100)
    np.random.seed(100)
    random.seed(100)
    base_params = params.copy()

    runs = []
    for N in episode_counts:
        run_params = base_params.copy()
        run_params["N_episodes"] = N

        actor, critic, rewards, steps = train_ddpg(run_params)

        runs.append({
            "label": f"N={N}",
            "rewards": rewards
        })
        del actor, critic
        torch.cuda.empty_cache()

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
    torch.manual_seed(100)
    np.random.seed(100)
    random.seed(100)
    base_params = params.copy()

    runs = []
    for buff in buffer_counts:
        run_params = base_params.copy()
        run_params["buffer_size"] = buff

        actor, critic, rewards, steps = train_ddpg(run_params)

        runs.append({
            "label": f"buffer_size={buff}",
            "rewards": rewards
        })
        del actor, critic
        torch.cuda.empty_cache()

    if save:
        torch.save(runs, "buffersize_runs.pth")

    plot_reward_sweep(
        runs,
        base_params["n_ep_running_average"],
        f"Effect of buffer size", 
        size, 
        save
    )

if __name__ == "__main__":
    torch.cuda.empty_cache()
    params = {
        "lr_actor": 5e-5,
        "lr_critic": 5e-4,
        "grad_clip": 1.0,
        "update_actor_every": 2,
        "tau":1e-3,
        "batch_size": 64,
        "buffer_size": 30000,
        "discount_factor": 0.99,
        "n_ep_running_average": 50,
        "N_episodes": 400,
        "sigma_noise": 0.2,
        "mu_noise": 0.15,
        "sigma_decay": 0.9995,
        "start_steps": 10000,
        "start_update": 1000
    }
    
    params_chunking = {
        "lr_actor": 5e-5,
        "lr_critic": 5e-4,
        "grad_clip": 1.0,
        "train_every": 40,
        "update_actor_every": 1,
        "tau":1e-3,
        "batch_size": 64,
        "min_buf": 30000,
        "buffer_size": 100000,
        "discount_factor": 0.99,
        "n_ep_running_average": 50,
        "N_episodes": 300,
        "sigma_noise": 0.2,
        "mu_noise": 0.15,
    }
    normal_run = False
    if normal_run:
        save(params)
        actor, critic, rewards, steps = train_ddpg(params)
        
        save(rewards, "rewards.pth")
        save(steps, "steps.pth")

        save_model(actor, filename="neural-network-2-actor.pth")
        save_model(critic, filename="neural-network-2-critic.pth")
        #rewards = load("rewards.pth")
        #steps = load("steps.pth")

        plot_rewards_steps(rewards, steps, save=True)

    chunking_run = False
    if chunking_run:
        save(params_chunking)
        actor, critic, rewards, steps = train_ddpg_chunking(params_chunking)
        
        save(rewards, "rewards-chunking.pth")
        save(steps, "steps-chunking.pth")

        save_model(actor, filename="neural-network-2-actor-chunking.pth")
        save_model(critic, filename="neural-network-2-critic-chunking.pth")
        #rewards = load("rewards.pth")
        #steps = load("steps.pth")

        plot_rewards_steps(rewards, steps, save=True)

    run_comparison = False
    if run_comparison:
        torch.manual_seed(100)
        np.random.seed(100)
        random.seed(100)

        discount_factors = [0.99, 0.5, 1]
        run_discount_factor_comparison(params, discount_factors, save=True)

        buffer_counts = [30000, 100000, 1000000]
        run_buffersize_comparison(params, buffer_counts, save=True)

        episode_counts = [200, 400, 600]
        run_N_episodes_comparison(params, episode_counts, save=True)

    plot3d = False
    if plot3d:
        critic = load_model("neural-network-2-critic.pth")
        actor = load_model("neural-network-2-actor.pth")
        plot3d_grid(actor, critic, save=True)

    compare_with_random = True
    if compare_with_random:
        actor = load_model("neural-network-2-actor.pth")
        actor.eval()

        env = gym.make('LunarLanderContinuous-v3')
        device = get_device()

        env.reset()
        
        m = len(env.action_space.high)
        random_agent = RandomAgent(m)

        # noise not being used for eval
        ddpg_agent = DDPGAgent(m, actor, device, noise = None)

        check_solution(env, random_agent)
        check_solution(env, ddpg_agent)

