# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random

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

    # Reward values 
    STEP_REWARD = -1
    GOAL_REWARD = 1
    KEY_REWARD = 0
    IMPOSSIBLE_REWARD = -1e6
    MINOTAUR_REWARD = -1

    


    def __init__(self, maze):
        """ Constructor of the environment Maze.
        """
        self.key_pos                  = (0, 7)
        self.with_key                 = 1
        self.allow_mino_stay_still    = 1
        self.move_towards             = 0
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
        
        return states, map

    def __move(self, state, action):               
        """ Makes a step in the maze, given a current position and an action. 
            If the action STAY or an inadmissible action is used, the player stays in place.
        
            :return list of tuples next_state: Possible states ((x,y), (x',y')) on the maze that the system can transition to.
        """        
        if self.states[state] == 'Eaten' or self.states[state] == 'Win': # In these states, the game is over
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
   
                        if (self.with_key):
                            
                            if (self.states[state][2] == 1):
                                states.append('Win')
                            
                            else:
                                states.append(((self.states[state][0][0], self.states[state][0][1]), (rows_minotaur[i], cols_minotaur[i]), self.states[state][2]))
                        
                        else:
                            states.append('Win')

                    else:     # The player remains in place, the minotaur moves randomly
                        
                        if self.with_key:
                            states.append(((self.states[state][0][0], self.states[state][0][1]), (rows_minotaur[i], cols_minotaur[i]), self.states[state][2]))
                        else:
                            states.append(((self.states[state][0][0], self.states[state][0][1]), (rows_minotaur[i], cols_minotaur[i])))
                        
                return states
          
            else: # The action is possible, the player and the minotaur both move
                states = []
                for i in range(len(rows_minotaur)):
                
                    if (row_player == rows_minotaur[i]) and (col_player == cols_minotaur[i]):                          # TODO: We met the minotaur
                        states.append('Eaten')
                    
                    elif (self.maze[row_player, col_player] == 2):                          # TODO:We are at the exit state, without meeting the minotaur
                        
                        if self.with_key:
                            if (self.states[state][2] == 1):
                                states.append('Win')
                            else:
                                states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]), self.states[state][2]))
                    
                        else:
                            states.append('Win')#states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]), self.states[state][2]))
                    
                    elif (self.with_key) and (self.states[state][2] == 0) and (self.states[state][0] == self.key_pos):
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]), 1))
                    
                    elif self.with_key: # The player moves, the minotaur moves randomly
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]), self.states[state][2]))

                    else:
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i])))
                if not states:
                        print(self.states[state])
              
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
    
    def __transitions_mixed(self, prob_toward_player=0.65):
        """Minotaur moves toward player with prob_toward_player, 
        otherwise uniformly among valid moves"""
        
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)
        
        for state_id in self.states:
            # Handle terminal states
            if self.states[state_id] in ['Eaten', 'Win']:
                for action in self.actions:
                    transition_probabilities[state_id, state_id, action] = 1.0
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

    
    def __transition_probabilities_toward_player(self):
        """ Computes the transition probabilities with the minotaur moving towards the player.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)
        
        for state_id in self.states:
            # Handle terminal states
            if self.states[state_id] == 'Eaten':
                for action in self.actions:
                    transition_probabilities[state_id, state_id, action] = 1.0
                continue
                
            if self.states[state_id] == 'Win':
                for action in self.actions:
                    transition_probabilities[state_id, state_id, action] = 1.0
                continue
            
            for action in self.actions:
                # Get current positions
                player_pos = self.states[state_id][0]
                mino_pos = self.states[state_id][1]
                has_key = self.states[state_id][2] if self.with_key else None
                
                # Compute next player position
                row_player = player_pos[0] + self.actions[action][0]
                col_player = player_pos[1] + self.actions[action][1]
                
                # Check if player action is valid
                impossible_action = (row_player == -1) or \
                                (row_player == self.maze.shape[0]) or \
                                (col_player == -1) or \
                                (col_player == self.maze.shape[1]) or \
                                (self.maze[row_player, col_player] == 1)
                
                if impossible_action:
                    next_player_pos = player_pos  # Player stays in place
                else:
                    next_player_pos = (row_player, col_player)
                
                # Update key status if applicable
                if self.with_key:
                    if has_key == 0 and next_player_pos == self.key_pos:
                        next_has_key = 1
                    else:
                        next_has_key = has_key
                
                # Get all valid minotaur moves (cannot go out of bounds)
                valid_mino_moves = []
                for mino_act in [self.MOVE_LEFT, self.MOVE_RIGHT, self.MOVE_UP, self.MOVE_DOWN]:
                    test_pos = (mino_pos[0] + self.actions[mino_act][0],
                            mino_pos[1] + self.actions[mino_act][1])
                    if (0 <= test_pos[0] < self.maze.shape[0] and 
                        0 <= test_pos[1] < self.maze.shape[1]):
                        valid_mino_moves.append((mino_act, test_pos))
                
                # Determine minotaur's move towards the player's NEXT position
                preferred_action = self.move_toward_player(mino_pos, next_player_pos)
                preferred_pos = (mino_pos[0] + self.actions[preferred_action][0],
                            mino_pos[1] + self.actions[preferred_action][1])
                
                # Check if preferred move is valid
                preferred_valid = (0 <= preferred_pos[0] < self.maze.shape[0] and 
                                0 <= preferred_pos[1] < self.maze.shape[1])
                
                if preferred_valid:
                    next_mino_pos = preferred_pos
                else:
                    # Choose the best alternative from valid moves
                    # Pick the move that minimizes distance to player
                    best_dist = float('inf')
                    next_mino_pos = mino_pos  # fallback
                    for mino_act, test_pos in valid_mino_moves:
                        dist = abs(test_pos[0] - next_player_pos[0]) + abs(test_pos[1] - next_player_pos[1])
                        if dist < best_dist:
                            best_dist = dist
                            next_mino_pos = test_pos
                
                # Determine the resulting state
                if next_player_pos == next_mino_pos:
                    next_state = 'Eaten'
                elif self.maze[next_player_pos[0], next_player_pos[1]] == 2:
                    next_state = 'Win'
                else:
                    if self.with_key:
                        next_state = (next_player_pos, next_mino_pos, next_has_key)
                    else:
                        next_state = (next_player_pos, next_mino_pos)
                
                # Set transition probability
                next_state_id = self.map[next_state]
                transition_probabilities[state_id, next_state_id, action] = 1.0
        
        return transition_probabilities
    
    def __transitions(self):
        probability_move_toward_player = 0.65
        probability_move_uniform = 0.35
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

                else:                
                    next_states = self.__move(s,a)
                    next_s = next_states[0] # The reward does not depend on the next position of the minotaur, we just consider the first one
                    
                    if self.states[s][0] == next_s[0] and a != self.STAY: # The player hits a wall
                        rewards[s, a] = self.IMPOSSIBLE_REWARD

                    elif (self.with_key) and (self.states[s][2] == 0) and (next_states[0][2] == 1): # the player picks up the key
                        #print("KEY REWARD")
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
                
                #next_states = self.__move(s, a) 
                #next_s = random.choice(next_states)
                
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

            #next_states = self.__move(s, int(policy[s])) # Move to next state given the policy and the current state
            #next_s = random.choice(next_states)
            
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
                
                #next_states = self.__move(s, policy[s]) # Move to next state given the policy and the current state
                #next_s = random.choice(next_states)
                
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

        for sT
        go through all states, compute there rewards
        for T-1:
            go through all states s
            calculate all their rewards based on all actions
            for each action:
                sum over all ppossible next state, and take weighted sum of probabilities * previous rewards
            take the max

        for every t:T

            for every state s:
                for every action a:
                    compute V(s') = r(s,a) + sum_s' [p(s'|s,a)*V(s')]
                V(s,t) = argmax(V(s'))
                policy(s,t) = argmax(a)
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
    
    Algorithm:
    while delta < eps(1-lambda)/lambda:
        Vn+1 = L[Vn]
        for all states s:
            for all actions a:
                Q
        
    """
    S = env.n_states
    V, policy = np.zeros(S), np.zeros(S)

    delta = 10
    stop_thresh = epsilon*(1 - gamma) / gamma

    # transition probabilities = (S,N,A), V_n-1 = (S)
    while delta > stop_thresh:
        Q_table = env.rewards + gamma * np.einsum(("sna,n->sa"), env.transition_probabilities, V)
        max_idxs = np.argmax(Q_table, axis=1)
        V_next = Q_table[np.arange(S), max_idxs]
        policy = max_idxs
        delta = np.linalg.norm(V_next - V)
        V = V_next        
    return V, policy

def Q_learning(env, nr_of_episodes, start_pos):
    
    # initialize Q(S,A)
    Q = np.zeros((env.n_states, env.n_actions))
    path = []

    for k in range(nr_of_episodes):
        s = env.map[start]
        path.append(start)
        t = 0
        while (is_not_finished(s)):
            pass

def is_not_finished(state):
    return (state == "Eaten") or (state == "Win")



def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -1: LIGHT_RED, -2: LIGHT_PURPLE}
    
    rows, cols = maze.shape # Size of the maze
    fig = plt.figure(1, figsize=(cols, rows)) # Create figure of the size of the maze

    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create a table to color
    grid = plt.table(
        cellText = None, 
        cellColours = colored_maze, 
        cellLoc = 'center', 
        loc = (0,0), 
        edges = 'closed'
    )
    
    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    for i in range(0, len(path)):
        if path[i-1] != 'Eaten' and path[i-1] != 'Win':
            grid.get_celld()[(path[i-1][0])].set_facecolor(col_map[maze[path[i-1][0]]])
            grid.get_celld()[(path[i-1][1])].set_facecolor(col_map[maze[path[i-1][1]]])
        if path[i] != 'Eaten' and path[i] != 'Win':
            grid.get_celld()[(path[i][0])].set_facecolor(col_map[-2]) # Position of the player
            grid.get_celld()[(path[i][1])].set_facecolor(col_map[-1]) # Position of the minotaur
        display.display(fig)
        time.sleep(0.1)
        display.clear_output(wait = True)



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
    horizon =  50    # TODO: Finite horizon
    gamma = 1 - 1/50
    epsilon = 1e-5
    nr_of_episodes = 10

    print("Computing optimal policy...")
    
    # Solve the MDP problem with dynamic programming
    #V, policy = dynamic_programming(env, horizon)  
    #V, policy = value_iteration(env, gamma, epsilon)
    
    print("Simulating...")

    # Simulate the shortest path starting from position A
    #method = 'ValIter'
    #method = 'DynProg'
    
    if env.with_key:
        start = ((0,0), (6,5), 0)
    else:
        start  = ((0,0), (6,5))

    Q = Q_learning(env, nr_of_episodes, start)
        
    NUM_SIMULATIONS = 10000
    NUM_WINS = 0
    final_states = []
    
    """
    for i in range(NUM_SIMULATIONS):
        path = env.simulate(start, policy, method)[0]
        NUM_WINS += (path[-1] == 'Win')
    #animate_solution(maze, path)

    print(f"Probability of winning: {NUM_WINS / NUM_SIMULATIONS}")
    #env.show()
    print(f"Final state: {path[-1]} ")
    print(path[-1])
    """


