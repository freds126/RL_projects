# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random
import os
import shutil

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }
    sparse_rewards = True
    # Reward values 
    if sparse_rewards:
        STEP_REWARD = 0
        GOAL_REWARD = 1
        KEY_REWARD = 0
        IMPOSSIBLE_REWARD = -1e6
        MINOTAUR_REWARD = 0
    else:
        STEP_REWARD = -1
        GOAL_REWARD = 100
        KEY_REWARD = 100
        IMPOSSIBLE_REWARD = -1e6
        MINOTAUR_REWARD = -100

    


    def __init__(self, maze):
        """ Constructor of the environment Maze.
        """
        self.key_pos                  = (0, 7)
        self.with_key                 = 1
        self.allow_mino_stay_still    = 0
        self.move_towards             = 1
        self.move_towards_probability = 0.35
        self.maze                     = maze
        self.actions                  = self.__actions()
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards()

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions

    def __states(self):
        
        states = dict()
        map = dict()
        s = 0

        
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        if self.maze[i,j] != 1:
                            if self.with_key:
                                for has_key in range(2):
                                    states[s] = ((i,j), (k,l), has_key)
                                    map[((i,j), (k,l), has_key)] = s
                                    s += 1
                            else:
                                states[s] = ((i,j), (k,l))
                                map[((i,j), (k,l))] = s
                                s += 1
        
        states[s] = 'Eaten'
        map['Eaten'] = s
        s += 1
        
        states[s] = 'Win'
        map['Win'] = s
        s += 1

        states[s] = 'Terminal'
        map['Terminal'] = s
        
        return states, map

    def __move(self, state, action):               
        """ Makes a step in the maze, given a current position and an action. 
            If the action STAY or an inadmissible action is used, the player stays in place.
        
            :return list of tuples next_state: Possible states ((x,y), (x',y')) on the maze that the system can transition to.
        """        
        if self.states[state] == 'Eaten' or self.states[state] == 'Win': # In these states, the game is over
            return ['Terminal'] 
            #return[self.states[state]]
        
        if self.states[state] == 'Terminal':
            return [self.states[state]]
        
        else: # Compute the future possible positions given current (state, action)
            row_player = self.states[state][0][0] + self.actions[action][0] # Row of the player's next position 
            col_player = self.states[state][0][1] + self.actions[action][1] # Column of the player's next position 
            
            # Is the player getting out of the limits of the maze or hitting a wall?
            impossible_action_player = (row_player == -1) or \
                                        (row_player == self.maze.shape[0]) or \
                                        (col_player == -1) or \
                                        (col_player == self.maze.shape[1]) or \
                                        (self.maze[row_player, col_player] == 1)
            
            # Possible moves for the Minotaur
            actions_minotaur = [[0, -1], [0, 1], [-1, 0], [1, 0]]
            if self.allow_mino_stay_still:
                actions_minotaur.append([0, 0])
            rows_minotaur, cols_minotaur = [], []
            for i in range(len(actions_minotaur)):
                # Is the minotaur getting out of the limits of the maze?
                impossible_action_minotaur = (self.states[state][1][0] + actions_minotaur[i][0] == -1) or \
                                             (self.states[state][1][0] + actions_minotaur[i][0] == self.maze.shape[0]) or \
                                             (self.states[state][1][1] + actions_minotaur[i][1] == -1) or \
                                             (self.states[state][1][1] + actions_minotaur[i][1] == self.maze.shape[1])
            
                if not impossible_action_minotaur:
                    rows_minotaur.append(self.states[state][1][0] + actions_minotaur[i][0])
                    cols_minotaur.append(self.states[state][1][1] + actions_minotaur[i][1])  
          

            # Based on the impossiblity check return the next possible states.
            if impossible_action_player: # The action is not possible, so the player remains in place
                states = []
                for i in range(len(rows_minotaur)):
                    
                    if (self.states[state][0][0] == rows_minotaur[i]) and (self.states[state][0][1] == cols_minotaur[i]):
                        states.append('Eaten')
                        
                    elif self.maze[self.states[state][0][0], self.states[state][0][1]] == 2:                           # We are at the exit state, without meeting the minotaur
   
                        if (self.with_key): # if state with key
                            
                            if (self.states[state][2] == 1): # if player has key
                                states.append('Win')
                            
                            else: # state is position like usual
                                states.append(((self.states[state][0][0], self.states[state][0][1]), (rows_minotaur[i], cols_minotaur[i]), self.states[state][2]))
                        
                        else: # if state doesnt have key, player wins instantly
                            states.append('Win')

                    else: # The player remains in place, the minotaur moves randomly
                        
                        if self.with_key: # update state - with key and positions
                            states.append(((self.states[state][0][0], self.states[state][0][1]), (rows_minotaur[i], cols_minotaur[i]), self.states[state][2]))
                        
                        else: # update state with only positions
                            states.append(((self.states[state][0][0], self.states[state][0][1]), (rows_minotaur[i], cols_minotaur[i])))
                        
                return states
          
            else: # The action is possible, the player and the minotaur both move
                states = []
                for i in range(len(rows_minotaur)):
                
                    if (row_player == rows_minotaur[i]) and (col_player == cols_minotaur[i]):                          # TODO: We met the minotaur
                        states.append('Eaten')

                    elif (self.maze[row_player, col_player] == 2):                          # TODO:We are at the exit state, without meeting the minotaur
                        
                        if self.with_key:
                            
                            if (self.states[state][2] == 1): # if player has key
                                states.append('Win')

                            else: # append state as normal
                                states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]), self.states[state][2]))
                    
                        else:
                            states.append('Win') # if state doesnt include key
                    
                    elif (self.with_key) and (self.states[state][2] == 0) \
                        and (row_player == self.key_pos[0]) \
                        and (col_player == self.key_pos[1]):    # if player moves to the key position

                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]), 1))
                   
                    # The player moves, the minotaur moves randomly

                    elif self.with_key: # if key is part of the state
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]), self.states[state][2]))

                    else: # state doesnt include key
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i])))

                return states
        
    def __transitions_uniform(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)
        
        for state_id in self.states:
            for action in self.actions:
                possible_future_states = self.__move(state_id, action)
                num_pos_future_states = len(possible_future_states)
                
                # Count occurrences of each unique state
                state_counts = {}
                for pos_future_state in possible_future_states:
                    pos_future_state_id = self.map[pos_future_state]
                    state_counts[pos_future_state_id] = state_counts.get(pos_future_state_id, 0) + 1
                
                # Assign probabilities based on counts
                for next_state_id, count in state_counts.items():
                    transition_probabilities[state_id, next_state_id, action] = count / num_pos_future_states
        
        return transition_probabilities
    
    def __transitions_mixed(self):
        """Minotaur moves toward player with prob_toward_player, 
        otherwise uniformly among valid moves"""
        prob_toward_player = self.move_towards_probability
        
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)
        
        for state_id in self.states:

            # handle terminal state
            if self.states[state_id] == 'Terminal':
                for action in self.actions:
                    transition_probabilities[state_id, state_id, action] = 1.0
                continue

            # Handle win/lose states
            if self.states[state_id] in ['Eaten', 'Win']:
                for action in self.actions:
                    transition_probabilities[state_id, self.map['Terminal'], action] = 1.0
                continue
            
            for action in self.actions:
                player_pos = self.states[state_id][0]
                mino_pos = self.states[state_id][1]
                has_key = self.states[state_id][2] if self.with_key else None
                
                # Calculate next player position
                row_player = player_pos[0] + self.actions[action][0]
                col_player = player_pos[1] + self.actions[action][1]
                
                impossible_action = (row_player < 0 or row_player >= self.maze.shape[0] or
                                col_player < 0 or col_player >= self.maze.shape[1] or
                                self.maze[row_player, col_player] == 1)
                
                next_player_pos = player_pos if impossible_action else (row_player, col_player)
                
                # Update key status
                if self.with_key:
                    next_has_key = 1 if (has_key == 0 and next_player_pos == self.key_pos) else has_key
                
                # Get ALL valid minotaur moves
                valid_mino_moves = []
                possible_mino_moves = [self.MOVE_LEFT, self.MOVE_RIGHT, self.MOVE_UP, self.MOVE_DOWN]
                
                if self.allow_mino_stay_still:
                    possible_mino_moves.append(self.STAY)

                for mino_act in possible_mino_moves:
                    test_pos = (mino_pos[0] + self.actions[mino_act][0],
                            mino_pos[1] + self.actions[mino_act][1])
                    if (0 <= test_pos[0] < self.maze.shape[0] and 
                        0 <= test_pos[1] < self.maze.shape[1]):
                        valid_mino_moves.append(test_pos)
                
                # Determine preferred move toward player
                preferred_action = self.move_toward_player(mino_pos, next_player_pos)
                preferred_pos = (mino_pos[0] + self.actions[preferred_action][0],
                            mino_pos[1] + self.actions[preferred_action][1])
                
                # If preferred move is invalid, find best alternative
                if not (0 <= preferred_pos[0] < self.maze.shape[0] and 
                        0 <= preferred_pos[1] < self.maze.shape[1]):
                    best_dist = float('inf')
                    for test_pos in valid_mino_moves:
                        dist = abs(test_pos[0] - next_player_pos[0]) + abs(test_pos[1] - next_player_pos[1])
                        if dist < best_dist:
                            best_dist = dist
                            preferred_pos = test_pos
                
                # Assign probabilities to all valid next states
                uniform_prob = (1 - prob_toward_player) / len(valid_mino_moves)
                
                for mino_next_pos in valid_mino_moves:
                    # Determine resulting state
                    if next_player_pos == mino_next_pos:
                        next_state = 'Eaten'
                    elif self.maze[next_player_pos[0], next_player_pos[1]] == 2:
                        if self.with_key:
                            next_state = 'Win' if (self.states[state_id][2] == 1) else (next_player_pos, mino_next_pos, next_has_key)
                        else:
                            next_state = 'Win'
                    else:
                        next_state = (next_player_pos, mino_next_pos, next_has_key) if self.with_key else (next_player_pos, mino_next_pos)
                    
                    next_state_id = self.map[next_state]
                    
                    # Add probability: uniform component for all moves
                    transition_probabilities[state_id, next_state_id, action] += uniform_prob
                    
                    # Add extra probability if this is the preferred move
                    if mino_next_pos == preferred_pos:
                        transition_probabilities[state_id, next_state_id, action] += prob_toward_player
        
        return transition_probabilities
    
    def __transitions(self):

        if self.move_towards:
            return self.__transitions_mixed()
        else:
            return self.__transitions_uniform()

    def __rewards(self):
        
        """ Computes the rewards for every state action pair """

        rewards = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                
                if self.states[s] == 'Eaten': # The player has been eaten
                    rewards[s, a] = self.MINOTAUR_REWARD
                
                elif self.states[s] == 'Win': # The player has won
                    rewards[s, a] = self.GOAL_REWARD
                
                elif self.states[s] == 'Terminal':
                    rewards[s, a] = 0

                else:                                  
                    next_states = self.__move(s,a)
                    next_s = next_states[0] # The reward does not depend on the next position of the minotaur, we just consider the first one
                    
                    if self.states[s][0] == next_s[0] and a != self.STAY: # The player hits a wall
                        rewards[s, a] = self.IMPOSSIBLE_REWARD

                    elif (self.with_key and self.states[s][2] == 0 and 
                        next_s[2] == 1): # the player picks up the key
                        rewards[s,a] = self.KEY_REWARD
      
                    else: # Regular move
                        rewards[s, a] = self.STEP_REWARD

        return rewards

    def move_toward_player(self, mino_pos, player_pos):
        # returns the up,down,left,right direction most toward the player from the minotaur

        dx = player_pos[0] - mino_pos[0]
        dy = player_pos[1] - mino_pos[1]

        if np.abs(dx) > np.abs(dy):
            return self.MOVE_DOWN if dx > 0 else self.MOVE_UP
        else:
            return self.MOVE_RIGHT if dy > 0 else self.MOVE_LEFT


    def simulate(self, start, policy, method):
        
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        
        if method == 'DynProg':
            horizon = policy.shape[1] # Deduce the horizon from the policy shape
            t = 0 # Initialize current time
            s = self.map[start] # Initialize current state 
            path.append(start) # Add the starting position in the maze to the path
            
            while t < horizon - 1:
                a = policy[s, t] # Move to next state given the policy and the current state
                
                # Get the probability distribution over next states
                next_state_probs = env.transition_probabilities[s, :, int(a)]

                # Sample next state according to the probability distribution
                next_s_id = np.random.choice(env.n_states, p=next_state_probs)
                next_s = env.states[next_s_id]

                path.append(next_s)  # Add the next state to the path

                path.append(next_s) # Add the next state to the path
                t +=1 # Update time and state for next iteration
                s = self.map[next_s]
                
        if method == 'ValIter': 
            t = 1 # Initialize current state, next state and time
            s = self.map[start]
            path.append(start) # Add the starting position in the maze to the path
            
            # Get the probability distribution over next states
            next_state_probs = env.transition_probabilities[s, :, int(policy[s])]

            # Sample next state according to the probability distribution
            next_s_id = np.random.choice(env.n_states, p=next_state_probs)
            next_s = env.states[next_s_id]
            
            path.append(next_s) # Add the next state to the path
            
            horizon = 50                           # Question e
            # Loop while state is not the goal state
            while s != next_s and t <= horizon:
                s = self.map[next_s] # Update state
                
                # Get the probability distribution over next states
                next_state_probs = env.transition_probabilities[s, :, int(policy[s])]

                # Sample next state according to the probability distribution
                next_s_id = np.random.choice(env.n_states, p=next_state_probs)
                next_s = env.states[next_s_id]
                
                path.append(next_s) # Add the next state to the path
                t += 1 # Update time for next iteration
        
        return [path, horizon] # Return the horizon as well, to plot the histograms for the VI



    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)



def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T

        tran_prob (S',S,A) = p(s'|s,a)
    """

    # initialize matrices
    S = env.n_states
    V = np.zeros((S, horizon))
    policy = np.zeros((S, horizon))

    T = horizon - 1

    # compute first round rewards: R(s,a)
    V[:, T] = np.max(env.rewards, axis=1)
    policy[:, T] = np.argmax(env.rewards, axis=1)
    
    # transition probabilities= (S,N,A), V[:,T] = (N,1)
    for t in range(T-1, -1, -1):
        future_rewards = np.einsum("sna,n->sa", env.transition_probabilities, V[:, t+1])
        Q = env.rewards + future_rewards

        V[:, t] = np.max(Q, axis=1)
        policy[:, t] = np.argmax(Q, axis=1)
    return V, policy

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
        
    """
    S = env.n_states
    V, policy = np.zeros(S), np.zeros(S)
    start = ((0,0), (6, 5))
    delta = 10
    stop_thresh = epsilon*(1 - gamma) / gamma
    V_starts = []

    # transition probabilities = (S,N,A), V_n-1 = (S)
    while delta > stop_thresh:
        Q_table = env.rewards + gamma * np.einsum(("sna,n->sa"), env.transition_probabilities, V)
        max_idxs = np.argmax(Q_table, axis=1)
        V_next = Q_table[np.arange(S), max_idxs]
        policy = max_idxs
        delta = np.linalg.norm(V_next - V)
        V = V_next        
        V_starts.append(V[env.map[start]])

    return V, policy, V_starts


def Q_learning(env, Q, nr_of_episodes, start_pos, epsilon, gamma, debug=False):
    print("Q-Learning....")
    # initialize Q(S,A)
    T = 50
    S = env.n_states
    A = env.n_actions
    Q = np.ones((S, A))

    visited_counts = np.zeros((S, A))
    total_reward_list = []
    avg_reward_list = []
    V_start_list = []

    for k in range(nr_of_episodes):
        total_reward = 0
        t = 0

        if np.random.random() < 0.5:
            s = np.random.choice(S-3) # exclude terminal and win/lose states

        else:
            s = env.map[start_pos] 

        while t < T:
            # select action epsilon-greedily
            if np.random.random() < epsilon:
                action = np.random.choice(A)
            else:
                action = np.argmax(Q[s,:])

            # observe reward
            reward = env.rewards[s, action]
            total_reward += reward
            
            next_state_probs = env.transition_probabilities[s, :, int(action)]

            # Sample next state according to the probability distribution
            next_s_id = np.random.choice(S, p=next_state_probs)
            next_s = env.states[next_s_id]

            # calculate step size
            visited_counts[s, action] += 1
            n = visited_counts[s, action]
            
            #step = min(0.5, 10.0 / (10.0 + n))
            step = 1/(n)**(2/3)

            # if episode is over, next_V should be zero
            if is_finishedQ(next_s):
                Q[s, action] = Q[s, action] + step * (reward - Q[s, action])
                break
            
            # next V max of Q over all possible actions
            next_V = np.max(Q[next_s_id, :])

            # update Q
            Q[s, action] = Q[s, action] + step * (reward + gamma * next_V - Q[s, action]) 
            
            s = next_s_id
            t += 1

        # epsilon decay
        epsilon = max(0.01, epsilon * 0.99995) 

        # saves average rewards 
        if (k + 1) % 1000 == 0:
            avg_reward = np.mean(total_reward_list[-1000:])
            avg_reward_list.append(avg_reward)
            
            if debug:
                print(f"Episode {k+1}/{nr_of_episodes}, Avg Reward (last 1000): {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

        total_reward_list.append(total_reward)
        V_start_list.append(np.max(Q[env.map[start_pos], :]))

    policy = np.argmax(Q, axis=1)
        
    return Q, policy, avg_reward_list, V_start_list

def SARSA(env, Q, nr_of_episodes, start_pos, epsilon, gamma, step_decay=0.5, eps_decay=0.9, debug=False):
    print("SARSA.........")
    
    T = 50
    S = env.n_states
    A = env.n_actions

    visited_counts = np.zeros((S, A))
    total_reward_list = []
    avg_reward_list = []
    V_start_list = []   

    for k in range(1, nr_of_episodes):
        total_reward = 0
        t = 0

        if np.random.random() < 0.5:
            s = np.random.choice(S-3) # exclude terminal and win/lose states

        else:
            s = env.map[start_pos] 

        # initialize action epsilon-greedily
        if np.random.random() < epsilon:
            action = np.random.choice(A)
        else:
            action = np.argmax(Q[s,:])

        while t < T:
            
            # observe reward
            reward = env.rewards[s, action]
            total_reward += reward
            
            next_state_probs = env.transition_probabilities[s, :, int(action)]

            # Sample next state according to the probability distribution
            next_s_id = np.random.choice(S, p=next_state_probs)
            next_s = env.states[next_s_id]

            # calculate step size
            visited_counts[s, action] += 1
            n = visited_counts[s, action]
            
            #step = min(0.5, 10.0 / (10.0 + n))
            step = 1/((n)**(step_decay))

            # if episode is over, next_V should be zero
            if is_finishedSARSA(next_s):
                reward = env.rewards[next_s_id, 0] # doesnt matter which action, will always return GOAL_REWARD or MINOTAUR_REWARD
                Q[s, action] = Q[s, action] + step * (reward - Q[s, action])
                break

            if  np.random.random() < epsilon:
                next_action = np.random.choice(A)
            else:
                next_action = np.argmax(Q[next_s_id, :])

            # update Q
            Q[s, action] = Q[s, action] + step * (reward + gamma * Q[next_s_id, next_action] - Q[s, action]) 
            
            s = next_s_id
            action = next_action
            t += 1

        # epsilon decay
        epsilon =  1/(k**(eps_decay))

        # saves average rewards 
        if k % 1000 == 0:
            avg_reward = np.mean(total_reward_list[-1000:])
            avg_reward_list.append(avg_reward)
            
            if debug:
                print(f"Episode {k}/{nr_of_episodes}, Avg Reward (last 1000): {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

        total_reward_list.append(total_reward)
        V_start_list.append(np.max(Q[env.map[start_pos], :]))

    policy = np.argmax(Q, axis=1)
        
    return Q, policy, avg_reward_list, V_start_list

def is_finishedQ(state):
    return state == "Terminal" 

def is_finishedSARSA(state):
    return (state == "Win" or state == "Eaten")

def animate_solution(maze, path, save_figs = False):
    if save_figs:
        # Create/reset directory for saving frames
        save_dir = "policy_frames"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)        # delete old frames
        os.makedirs(save_dir)               # new empty folder

        frame_id = 0
    # Map a color to each cell in the maze
    col_map = {
        0:  WHITE,        # empty cell
        1:  BLACK,        # wall
        2:  LIGHT_GREEN,  # exit
        -1: LIGHT_RED,    # minotaur
        -2: LIGHT_PURPLE  # player
    }
    
    rows, cols = maze.shape

    # Create figure and axis ONCE
    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Base colors for the maze
    colored_maze = [[col_map[maze[r, c]] for c in range(cols)] for r in range(rows)]

    # Create table ONCE
    grid = ax.table(
        cellText=None,
        cellColours=colored_maze,
        cellLoc='center',
        loc='center',
        edges='closed'
    )

    # Normalize cell sizes
    cells = grid.get_celld()
    for (r, c), cell in cells.items():
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / cols)

    # Show initial figure once
    display.display(fig)

    # Animate through the path
    for i in range(1, len(path)):
        from IPython.display import clear_output

        clear_output(wait=True)

        prev_state = path[i - 1]
        curr_state = path[i]

        # ---- Reset previous state's player/minotaur cells back to maze colors ----
        if isinstance(prev_state, tuple):
            # prev_state can be ((pr,pc), (mr,mc)) or ((pr,pc), (mr,mc), has_key)
            prev_player = prev_state[0]
            prev_mino   = prev_state[1]

            pr, pc = prev_player
            mr, mc = prev_mino

            cells[(pr, pc)].set_facecolor(col_map[maze[pr, pc]])
            cells[(mr, mc)].set_facecolor(col_map[maze[mr, mc]])

        # ---- Draw current state's player/minotaur, unless it's a terminal string ----
        if isinstance(curr_state, tuple):
            curr_player = curr_state[0]
            curr_mino   = curr_state[1]

            pr, pc = curr_player
            mr, mc = curr_mino

            cells[(pr, pc)].set_facecolor(col_map[-2])  # player
            cells[(mr, mc)].set_facecolor(col_map[-1])  # minotaur

        # Re-display the SAME figure
        display.display(fig)

        if save_figs:
            # Save each frame (overwrite numbering each run)
            fig.savefig(f"{save_dir}/frame_{frame_id:03d}.png", dpi=120, bbox_inches="tight")
            frame_id += 1
        time.sleep(0.1)

if __name__ == "__main__":

    # Description of the maze as a numpy array
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]])
    # With the convention 0 = empty cell, 1 = obstacle, 2 = exit of the Maze
    
    env = Maze(maze) # Create an environment maze
    S = env.n_states
    A = env.n_actions

    horizon =  50    # Finite horizon
    gamma = 1 - 1/horizon
    epsilon = 0.1
    nr_of_episodes = 50000

    print("Computing optimal policy...")

    if env.with_key:
        start = ((0,0), (6,5), 0)
    else:
        start  = ((0,0), (6,5))

    plot = False

    if plot:
        V_0s = []

        # Solve the MDP problem with dynamic programming

        for i in range(horizon):
            V, policy = dynamic_programming(env, i+1)
            V_0s.append(V[env.map[start], 0])
        
        Ts = np.arange(1, horizon + 1)

        plt.figure()
        plt.plot(Ts, V_0s, marker='o', linewidth=2)
        plt.xlabel("Horizon $T$", size = 22)
        plt.ylabel("Probability of exiting alive", size = 22)
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.title("Exit probability vs horizon", size =22)
        plt.tight_layout()
        
        #plt.savefig("win_prob_stay.png") 
        plt.show()

    #V, policy = dynamic_programming(env, horizon)
    #V, policy, V_starts = value_iteration(env, gamma, epsilon)
    
    # Simulate the shortest path starting from position A
    method = 'ValIter'
    #method = 'DynProg'

    Q = np.ones((S, A))
    #Q[env.map['Win'], :] = 100
    #Q[env.map['Eaten'], :] = -100
    #Q = np.load("pretrained_Q.npy")

    Q, policy, rewards, V_starts = Q_learning(env, Q, nr_of_episodes, start, epsilon, gamma)
    #Q, policy, rewards, V_starts = SARSA(env, Q, nr_of_episodes, start, epsilon, gamma, debug=False)
    #np.save("pretrained_Q", Q)

    #np.save("latest_policy", policy)
    #policy = np.load("latest_policy.npy")

    print("Simulating...")
    
    NUM_SIMULATIONS = 10
    NUM_WINS = 0
    final_states = []
    
    for i in range(NUM_SIMULATIONS):
        path = env.simulate(start, policy, method)[0]
        NUM_WINS += ('Win' in path)
        #animate_solution(maze, path, save_figs=False)
    
    plot_reward = True
    if plot_reward:
        print(len(V_starts))
        plt.figure()
        plt.plot(V_starts, marker='o', linewidth=2)
        plt.xlabel("Iterations $i$", size = 22)
        plt.ylabel("Value function $V(s_0)$", size = 22)
        #plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.title("$V(s_0)$ computed with VI", size = 22)
        plt.tight_layout()
        
        #plt.savefig("V_Start_VI_30.png") 
        plt.show()

    
    print(f"Probability of winning: {NUM_WINS / NUM_SIMULATIONS}")
    #print(f"V_start: {np.max(V[env.map[start]])}")



