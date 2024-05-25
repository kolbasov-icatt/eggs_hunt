import numpy as np
from itertools import product
from typing import List, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_environment(n_rooms, n_eggs_list, initial_p_list, decay_rate_list, finding_reward=10, not_finding_penalty=-2):
    
    """ 
        Create our environment returning the transition probabilities, the possible states, the actions and the total number of eggs
        
        Input:
        
        n_rooms: number of rooms 
        n_eggs_list: list with the number of eggs for each room
        initial_p_list: list of initial probability to find an egg in each room
        decay_rate_list: decay rate for the probability of findin next egg
        finding_reward: reward for each egg found
        not_finding_penalty: penalty for each step in which you don't find an egg
        
        Output:
        transition probabilities
        possible states
        actions
        total number of eggs
        
    """
    
    # Create the possible actions
    actions = ['search', 'stay'] + [f'go_to_room_{i+1}' for i in range(n_rooms)]
    
    # Create all possible combinations of egg counts for n_rooms with different maximum counts
    room_eggs = {}
    for i in range(n_rooms):
        max_eggs = n_eggs_list[i]
        for egg_counts in product(*(range(max_eggs + 1) for max_eggs in n_eggs_list)):
            state = (f'Room {i+1}',) + tuple(egg_counts)
            room_eggs[state] = set(actions)
    
    # states and actions to numbers
    states_dict = {idx: state for idx, state in enumerate(list(room_eggs.keys()))}
    actions_dict = {idx: action for idx, action in enumerate(actions)}
    
    # Invert the states_dict and actions_dict for easy lookup
    inv_states_dict = {v: k for k, v in states_dict.items()}
    inv_actions_dict = {v: k for k, v in actions_dict.items()}
    
    transitions = {}
    
    # initialize the state, the room, the egg count and the transition probabilities
    for current_state in room_eggs.keys():
        current_state_idx = inv_states_dict[current_state]
        transitions[current_state_idx] = {}
        current_room = current_state[0]
        egg_counts = list(current_state[1:])
        current_room_index = int(current_room.split()[1]) - 1
        
        # take an action 
        for action in actions:
            action_idx = inv_actions_dict[action]
            
            # action search try to find eggs in the current room
            if action == 'search':
                initial_p = initial_p_list[current_room_index]
                # each egg found in a room decrease the probability to find another in the same room
                decay_rate = decay_rate_list[current_room_index]
                
                # if there aren't still eggs not found in the room the probability to find one is 0 and the probability to not find one is 1
                if egg_counts[current_room_index] >= n_eggs_list[current_room_index]:
                    found_egg_prob = 0.0
                    no_egg_prob = 1.0
                # if there are still eggs compute the probability to find it basing on the decay rate and the number of eggs found in that room
                else:
                    found_egg_prob = max(0, initial_p * decay_rate ** egg_counts[current_room_index])
                    no_egg_prob = 1 - found_egg_prob
                
                # State if an egg is found
                found_egg_state = list(current_state)
                if found_egg_prob > 0:
                    found_egg_state[current_room_index + 1] += 1
                found_egg_state = tuple(found_egg_state)
                
                # State if no egg is found
                no_egg_state = current_state
                
                # Only add the found egg state if the probability is greater than 0
                found_egg_state_idx = inv_states_dict[found_egg_state]
                no_egg_state_idx = inv_states_dict[no_egg_state]
                
                if found_egg_prob > 0:
                    transitions[current_state_idx][action_idx] = [
                        (found_egg_state_idx, finding_reward, found_egg_prob),  # Found an egg
                        (no_egg_state_idx, not_finding_penalty, no_egg_prob)  # Did not find an egg
                    ]
                else:
                    transitions[current_state_idx][action_idx] = [
                        (no_egg_state_idx, not_finding_penalty, 1.0)  # Did not find an egg, no_egg_prob is 1
                    ]
            # this action make you stay in the same state    
            elif action == 'stay':
                transitions[current_state_idx][action_idx] = [
                    (current_state_idx, 0, 1.0)  # Staying in the same state with reward 0
                ]
             # this action make you change room   
            else:
                # Extract the target room from the action
                target_room_index = int(action.split('_')[-1]) - 1
                if target_room_index == current_room_index:
                    continue  # Skip transition to the same room
                target_room = f'Room {target_room_index + 1}'
                
                # Create the next state
                next_state = (target_room,) + tuple(egg_counts)
                next_state_idx = inv_states_dict[next_state]
                
                transitions[current_state_idx][action_idx] = [
                    (next_state_idx, -1, 1.0)
                ]
        total_eggs = sum(n_eggs_list)
    return transitions, states_dict, actions_dict, total_eggs


def create_random_policy(env) -> np.array:
    
    """ Creates a random policy for our environment 
        Returns: numpy 2D array (number of states, number of actions), where
                 each entry - the prob of choosing an action in a state
                 (equal probability for all possible actions in the state)
    """
    
    # initialize policy 
    policy = np.zeros((env.num_states, env.num_actions))
    # for each state select a random action with uniform probability between the possible ones
    for state in env.states:
        possible_actions = env.get_possible_actions(state)
        prob = 1 / len(possible_actions)
        policy[state, possible_actions] = prob  # setting the probabilities at once
    return policy

def custom_argmax(q_state: np.array, possible_actions: List[int]) -> int:
    """ Returns the action with the maximum Q-value from a list of possible actions."""

    sub_array = q_state[possible_actions]
    max_index_in_sub_array = np.argmax(sub_array)
    
    # Map this index back to the original array's indices
    max_index_in_original = possible_actions[max_index_in_sub_array]
    return max_index_in_original

def create_q_dataframe_for_rooms(env, Q):
     
    """
     Create a dataframe containing the Q values for each room
     
     Input:
        Q: state-action value function
        
     Output:
         Dataframe containing Q
    """
    
    # Dictionary to store DataFrames for each room
    room_dfs = {}

    # Extract maximum number of eggs for each room
    max_eggs = [env.get_state_name(state)[1:] for state in env.states]
    max_eggs = list(map(lambda x: max(x), zip(*max_eggs)))  # Get max eggs per room

    for room_idx in range(len(max_eggs)):
        # Get the name of the room
        room_name = f'Room {room_idx + 1}'
        
        # Initialize the DataFrame for this room
        room_states = [state for state in env.states if env.get_state_name(state)[0] == room_name]
        actions = [env.get_action_name(action) for action in env.actions]
        df = pd.DataFrame(index=room_states, columns=actions)
        
        # Populate the DataFrame with Q-values
        for state in room_states:
            for action in env.actions:
                action_name = env.get_action_name(action)
                df.loc[state, action_name] = Q[state, action]
        
        df.rename(index=lambda state: env.get_state_name(state), inplace=True)
        df = df.astype('float')
        # Store the DataFrame in the dictionary
        room_dfs[room_name] = df
    return room_dfs


# Plotting the DataFrame as a heatmap
def plot_heatmap_q(room_dfs, room: str, ):
    """ 
    Returns an heatmap of the action-value function for each room
    """
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(room_dfs[room], annot=True, cmap='viridis', cbar=True, fmt='.2f')
    plt.title(f'Q-values Heatmap for {room}')
    plt.xlabel('Actions')
    plt.ylabel('States')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()


def sample_action_eps_greedy_policy(env, state, Q, eps):
     
    """ Returns an epsilon greedy action taking as input the state, the state-action value funtion and the epsilon"""
    
    # take the possible actions from the considered state and calculate the best one basing on the state-action value funtion
    possible_actions = env.get_possible_actions(state)
    best_action = custom_argmax(Q[state], possible_actions)
    
    # initialize the action probabilitiews as an empty list and populate it basing on the epsilon greedy rule
    action_probabilities = []
    for action in possible_actions:
        if action == best_action:        
            action_probabilities.append(1 - eps + eps / len(possible_actions) )
        else:
            action_probabilities.append(eps / len(possible_actions))
    
    # take an action randomly from the possible ones with probability given by the action probabilites
    action = np.random.choice(possible_actions, p=action_probabilities)       
    return action

def compute_optimal_policy(env, Q):
    """ 
    Returns the optimal policy taking as input the state-action value function.
    """
    # initizialize the optimal policy
    optimal_policy = np.zeros((env.num_states, env.num_actions))
    
    for state in env.states:
        if env.is_terminal_state(state):
            optimal_policy[state, 1] = 1  # in terminal state you should "stay"
        
        # update the optimal policy in a greedy way
        else:
            best_action = custom_argmax(Q[state], env.get_possible_actions(state))
            for action in env.get_possible_actions(state):
                if action == best_action:
                    optimal_policy[state][action] = 1
    return optimal_policy

def linear_schedule(begin_value, end_value, begin_t, end_t=None, decay_steps=None):
    """Linear schedule, used for exploration epsilon in DQN agents."""

    decay_steps = decay_steps if end_t is None else end_t - begin_t

    def step(t):
        """Implements a linear transition from a begin to an end value."""
        frac = min(max(t - begin_t, 0), decay_steps) / decay_steps
        return (1 - frac) * begin_value + frac * end_value

    return step


def generate_episode(env,Q, num_timesteps, eps):
    """
    Given a state-action value function returns an episode.
    
    Args:
        env: a MDP enviornment
        Q: a state-action value function
        num_timesteps: number of timesteps in each episode
        eps: exploration rate
    
    Returns:
        episode
    """
    #initialize an epty list for storing the episode
    episode = []
    
    #initialize the initial state using the reset function
    state = env.reset()
    
    #then for each time step
    for t in range(num_timesteps):
        
        #select the action according to the epsilon-greedy policy
        action = sample_action_eps_greedy_policy(env, state, Q, eps)
        
        #perform the selected action and store the next state information
        reward, next_state, done = env.step(action)
        
        #store the state, action, reward in the episode list
        episode.append((state, action, reward))
        
        #if the next state is a final state then break the loop else update the next state to the current state
        if done:
            break
            
        state = next_state

    return episode


###################### FOR VFA ##############################################
def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    """Given the list of rewards output the final total return
        Args:
            rewards: list of rewards 
            gamma: decay rate

        Output:
        returns[::-1]: final cumulative return
    """
    # initialize the returns list and the return
    returns = []
    G = 0 
    
    # Loop over the rewards in reverse order
    for reward in reversed(rewards):
        # Update the return and store them in the returns list
        G = reward + gamma * G  
        returns.append(G)
      
    return returns[::-1]


def test_policy(env, policy, num_runs=10, n_episodes=100):
    total_returns = {}
    
    for n in range(1, num_runs + 1):
        total_return = 0
        env.reset()
        for ep in range(n_episodes):
            action = np.random.choice(env.actions, p=policy[env.current_state])
            reward, next_state, done = env.step(action)
            total_return += reward  # return for one run, but they can be different so we need to average them over many independent runs
            if done:
                print(f'Completed in {ep} episodes')
                break
        total_returns[f'Run {n}'] = (total_return, ep)

    average_return = np.mean([value[0] for k, value in total_returns.items()])
    return average_return, total_returns