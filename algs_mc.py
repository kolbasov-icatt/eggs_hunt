from utils import *
import numpy as np
from collections import defaultdict

def MC_onpolicy(env, policy, eps, num_timesteps, num_iterations):
    
    """
    Returns the optimal state-value function and the optimal policy.
    
    Args:
        env: a MDP enviornment
        eps: exploration rate
        num_timesteps: number of time steps in each episode
        num_iterations: number of episodes
    
    Returns:
        estimated state-value function
    """
    
    # initialize a state-action value function, the total return and a counter
    Q = np.zeros((env.num_states, env.num_actions))
    total_return = defaultdict(float)
    N = defaultdict(int)
    
    #for each episode
    for i in range(num_iterations):

        #we generate an episode starting from the Q function
        episode = generate_episode(env,Q, num_timesteps, eps)

        #get all the state-action pairs in the episode
        all_state_action_pairs = [(s, a) for (s,a,r) in episode]

        #store all the rewards obtained in the episode in the rewards list
        rewards = [r for (s,a,r) in episode]

        #for each step in the episode 
        for t, (state, action, reward) in enumerate(episode):

            #if the state-action pair is occurring for the first time in the episode
            if not (state, action) in all_state_action_pairs[0:t]:

                #compute the return R of the state-action pair as the sum of rewards
                R = sum(rewards[t:])

                #update total return of the state-action pair
                total_return[(state,action)] = total_return[(state,action)] + R

                #update the number of times the state-action pair is visited
                N[(state, action)] += 1

                #compute the Q value by just taking the average
                Q[(state,action)] = total_return[(state, action)] / N[(state, action)]
        
        #initialize the optimal policy
        optimal_policy = np.zeros((env.num_states, env.num_actions))
        # compute the optimal policy as a greedy policy
        for state in env.states:
            best_action = custom_argmax(Q[state], env.get_possible_actions(state))
            for action in env.get_possible_actions(state):
                if action == best_action:
                    optimal_policy[state][action] = 1
    
    return optimal_policy, Q