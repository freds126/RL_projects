# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal 
import matplotlib.pyplot as plt
from tqdm import trange
from PPO_agent import PPOAgent, RandomAgent
from ActorCriticNetworks import Actor, Critic
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
    
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

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0)

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
    #plt.close(fig)
    plt.show()

def train_ppo3(params):
    import gym
    import torch.optim as optim
    from tqdm import trange
    
    # Import and initialize Lunar Lander Environment
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()

    # Parameters
    N_episodes = params["N_episodes"]
    discount_factor = params["discount_factor"]
    n_ep_running_average = params["n_ep_running_average"]
    m = len(env.action_space.high)  # dimensionality of the action
    dim_state = len(env.observation_space.high)

    nr_of_epochs = params["nr_of_epochs"]  # M in Algorithm 3
    eps = params["eps"]  # ε for clipping

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Reward tracking
    episode_reward_list = []
    episode_number_of_steps = []

    # Initialize networks
    actor_net = Actor(dim_state, m).to(device)
    critic_net = Critic(dim_state, 1).to(device)

    # Optimizers
    optimizer_critic = optim.Adam(critic_net.parameters(), lr=params["lr_critic"])
    optimizer_actor = optim.Adam(actor_net.parameters(), lr=params["lr_actor"])

    # Agent initialization
    agent = PPOAgent(m, actor_net, device)

    # Training process
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset environment
        done, truncated = False, False
        state = env.reset()[0]
        total_episode_reward = 0.
        t = 0
        
        # Episode buffer
        states, actions, rewards = [], [], []
        
        # Collect trajectory (Algorithm 3, lines 6-10)
        while not (done or truncated):
            # Take action
            action = agent.forward(state)
            
            # Get next state and reward
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Append to buffer
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t += 1
        
        # Convert to tensors
        states = torch.as_tensor(np.array(states), device=device, dtype=torch.float32)
        actions = torch.as_tensor(np.array(actions), device=device, dtype=torch.float32)
        
        # Compute returns G_t (Algorithm 3, line 11)
        G = np.zeros(len(rewards), dtype=np.float32)
        running = 0
        for j in reversed(range(len(rewards))):
            running = rewards[j] + discount_factor * running
            G[j] = running
        G = torch.as_tensor(G, device=device, dtype=torch.float32)
        
        # Compute old policy log probabilities (Algorithm 3, line 13)
        with torch.no_grad():
            mu, std = actor_net(states)
            pi_old = Normal(mu, std)
            log_pi_old = pi_old.log_prob(actions).sum(-1)  # Sum over action dimensions
        
        # PPO training epochs (Algorithm 3, lines 14-17)
        for epoch in range(nr_of_epochs):
            # Get current value estimates
            V = critic_net(states).squeeze()
            
            # Compute advantages (Algorithm 3, line 12)
            # Note: We compute this inside the loop, but advantages are detached
            # so they don't change during the epoch
            with torch.no_grad():
                adv = G - V
                # Normalize advantages (common practice, improves stability)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            
            # Update Critic (Algorithm 3, line 15)
            V = critic_net(states).squeeze()  # Recompute for gradient
            loss_critic = nn.functional.mse_loss(G, V)
            
            optimizer_critic.zero_grad()
            loss_critic.backward()
            optimizer_critic.step()
            
            # Update Actor (Algorithm 3, line 16)
            mu, std = actor_net(states)
            pi = Normal(mu, std)
            log_pi = pi.log_prob(actions).sum(-1)
            
            # Compute probability ratio
            ratio = torch.exp(log_pi - log_pi_old)
            
            # Compute clipped objective
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
            loss_actor = -torch.min(surr1, surr2).mean()
            
            optimizer_actor.zero_grad()
            loss_actor.backward()
            optimizer_actor.step()

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Update progress bar
        if len(episode_reward_list) >= n_ep_running_average:
            avg_reward = np.mean(episode_reward_list[-n_ep_running_average:])
            avg_steps = np.mean(episode_number_of_steps[-n_ep_running_average:])
        else:
            avg_reward = np.mean(episode_reward_list)
            avg_steps = np.mean(episode_number_of_steps)
            
        EPISODES.set_description(
            f"Episode {i} - Reward/Steps: {total_episode_reward:.1f}/{t} - "
            f"Avg. Reward/Steps: {avg_reward:.1f}/{avg_steps:.0f}"
        )

    # Close environment
    env.close()
    
    return episode_reward_list, episode_number_of_steps

def train_ppo(params):
    # Import and initialize Mountain Car Environment
    env = gym.make('LunarLanderContinuous-v3')
    # If you want to render the environment while training run instead:
    # env = gym.make('LunarLanderContinuous-v3', render_mode = "human")
    torch.manual_seed(100)
    np.random.seed(100)

    env.reset()

    # Parameters
    N_episodes = params["N_episodes"]            # Number of episodes to run for training
    discount_factor = params["discount_factor"]         # Value of gamma
    n_ep_running_average = params["n_ep_running_average"]      # Running average of 50 episodes
    m = len(env.action_space.high) # dimensionality of the action
    dim_state = len(env.observation_space.high)

    nr_of_epochs = params["nr_of_epochs"]
    eps= params["eps"]
    kl_thresh = params["kl_threshold"]

    device = get_device()

    # Reward
    episode_reward_list = []  # Used to save episodes reward
    episode_number_of_steps = []

    # actor networks initialization
    actor_net = Actor(dim_state, m).to(device)

    # critc networks initialization
    critic_net = Critic(dim_state, 1).to(device)

    optimizer_critic = optim.Adam(critic_net.parameters(), lr=params["lr_critic"])
    optimizer_actor = optim.Adam(actor_net.parameters(), lr=params["lr_actor"])

    # DDPG Agent initialization
    agent = PPOAgent(m, actor_net, device)

    # Training process
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    global_t = 1
    avg_reward = 0
    #update_count = 0

    for i in EPISODES:
        # Reset enviroment data
        done, truncated = False, False
        state = env.reset()[0]
        total_episode_reward = 0.
        t = 0
        states, actions, rewards = [], [], []
        while not (done or truncated):
            # Take an action
            action = agent.forward(state)
         
            # Get next state and reward
            next_state, reward, done, truncated, _ = env.step(action)
            
            # append to buffer
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t+= 1
            global_t += 1
     
        states = torch.as_tensor(np.array(states), device=device, dtype=torch.float32)
        actions = torch.as_tensor(np.array(actions), device=device, dtype=torch.float32)
        G = np.zeros(len(rewards), dtype=np.float32)
    
        running = 0
        for j in reversed(range(len(rewards))):
            running = rewards[j] + discount_factor * running
            G[j] = running

        G = torch.as_tensor(G, device=device, dtype=torch.float32)
        
        with torch.no_grad():
            mu, var = actor_net(states)
            pi_old = Normal(mu, torch.sqrt(var))
            log_pi_old = pi_old.log_prob(actions).sum(-1)

        
        for j in range(nr_of_epochs):
            V = critic_net(states).squeeze()

            V_old = V.detach()
            V_clipped = V_old + torch.clamp(V - V_old, -eps, eps)
            loss_critic1 = nn.functional.mse_loss(G, V)
            loss_critic2 = nn.functional.mse_loss(G, V_clipped)
            loss_critic = torch.max(loss_critic1, loss_critic2)

            adv = (G - V).detach()
            adv = (adv )

            # critic update
            loss_critic = nn.functional.mse_loss(G, V)
            optimizer_critic.zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(critic_net.parameters(), max_norm=0.5)
            optimizer_critic.step()

            #  update actor
            mu, var = actor_net(states)
            pi = Normal(mu, torch.sqrt(var))
            log_pi = pi.log_prob(actions).sum(-1)

            prob_ratio = torch.exp(log_pi - log_pi_old)         # shape [T]
            
            kl = (log_pi_old - log_pi).mean()
            if kl > kl_thresh:
                break

            surr1 = prob_ratio * adv
            surr2 = torch.clamp(prob_ratio, 1-eps, 1+eps) * adv
            entropy = pi.entropy().mean()

            loss_actor = -(torch.min(surr1, surr2)).mean() 
            optimizer_actor.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=0.5)
            optimizer_actor.step()

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        avg_reward = running_average(episode_reward_list, n_ep_running_average)[-1]
        avg_steps = running_average(episode_number_of_steps, n_ep_running_average)[-1]
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,avg_reward, avg_steps
            ))
        
        if avg_reward > 280:
            print(f"Successfully trained agent to avg reward of {avg_reward} after {global_t} iterations")
            break

    # Close environment
    env.close()
    return critic_net, actor_net, episode_reward_list, episode_number_of_steps
    
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
        plt.savefig("epsilon_comp.png")
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
        mu, var = actor(states_t)                  # (N, act_dim)
        V = critic(states_t)              # (N, 1) or (N,)

    A = mu.clamp(-1, 1)
    V_grid = V.squeeze(-1).reshape(Ny, Nw)
    A_grid = A[:, 0].reshape(Ny, Nw)          # choose action dim

    # ---- VALUE PLOT ----
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Y, W, V_grid, linewidth=0, antialiased=True)

    ax.set_xlabel("y (height)", fontsize=size)
    ax.set_ylabel(r"$\omega$ (angle)  [rad]", fontsize=size)
    ax.set_zlabel(r"$V_{\omega}$", fontsize=size)
    ax.set_title(r"Value surface: $V_{\omega}(s(y,\omega))$", fontsize=size)

    if save:
        plt.savefig("V_3d_plot-3.png")
    plt.show()

    # ---- ACTION PLOT ----
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Y, W, A_grid, linewidth=0, antialiased=True)

    ax.set_xlabel("y (height)", fontsize=size)
    ax.set_ylabel(r"$\theta$ (angle)  [rad]", fontsize=size)
    ax.set_zlabel(r"$\pi_0(s)$", fontsize=size)
    ax.set_title(r"Actor action surface: $\mu_{\theta}(s)$", fontsize=size)

    if save:
        plt.savefig("action_3d_plot-3.png")
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

def run_discount_factor_comparison(params, discount_factors, save):
    torch.manual_seed(100)
    np.random.seed(100)
   
    base_params = params.copy()
    runs = []
    for discount_factor in discount_factors:
        run_params = base_params.copy()

        run_params["discount_factor"] = discount_factor
        filename = f"new_avg_reward_with_discount_{discount_factor}.png"

        actor, critic, epsiode_reward_list, epsiode_step_list = train_ppo(run_params)
        plot_rewards_steps(epsiode_reward_list, epsiode_step_list, filename, save)

        runs.append({
            "label": f"gamma={discount_factor}",
            "rewards": epsiode_reward_list
        })

        del actor, critic
        torch.cuda.empty_cache()
    if save:
        torch.save(runs, "discount_factors_runs.pth")

def run_epsilon_comparison(params, epsilons, size=20, save=False):
    torch.manual_seed(100)
    np.random.seed(100)
    base_params = params.copy()

    runs = []
    for eps in epsilons:
        run_params = base_params.copy()
        run_params["eps"] = eps

        actor, critic, rewards, steps = train_ppo(run_params)

        runs.append({
            "label": f"N={eps}",
            "rewards": rewards
        })
        del actor, critic
        torch.cuda.empty_cache()

    if save:
        torch.save(runs, "epsilon_runs.pth")

    plot_reward_sweep(
        runs,
        base_params["n_ep_running_average"],
        r"Effect of different $\varepsilon$ values", 
        size,
        save
    )


if __name__ == "__main__":
    params = {
        "lr_actor": 1e-5,
        "lr_critic": 1e-3,
        "discount_factor": 0.99,
        "n_ep_running_average": 50,
        "N_episodes": 1600,
        "nr_of_epochs": 10,
        "eps": 0.2,
        "kl_threshold": 0.015
    }
    normal_run = True
    if normal_run:
        save(params)
        critic, actor, rewards, steps = train_ppo(params)
        save_model(critic, "neural-network-3-critic.pth")
        save_model(actor, "neural-network-3-actor.pth")

        plot_rewards_steps(rewards, steps, save=True)

    run_comparison = False
    if run_comparison:
        torch.manual_seed(100)
        np.random.seed(100)

        discount_factors = [0.99, 0.7, 1]
        run_discount_factor_comparison(params, discount_factors, save=True)

        epsilons = [0.1, 0.3]
        run_epsilon_comparison(params, epsilons, save=True)

    plot3d = False
    if plot3d:
        critic = load_model("neural-network-3-critic.pth")
        actor = load_model("neural-network-3-actor.pth")
        
        plot3d_grid(actor, critic, save=True)

    compare_with_random = False
    if compare_with_random:
        actor = load_model("neural-network-3-actor.pth")
        actor.eval()

        env = gym.make('LunarLanderContinuous-v3')
        device = get_device()

        env.reset()
        
        m = len(env.action_space.high)
        random_agent = RandomAgent(m)

        # noise not being used for eval
        ddpg_agent = PPOAgent(m, actor, device)

        check_solution(env, random_agent)
        check_solution(env, ddpg_agent)