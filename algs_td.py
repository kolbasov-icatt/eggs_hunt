from utils import *
import numpy as np


def temploral_difference_policy_evaluation(env, policy, n_steps, gamma, alpha=0.05):
    
    """
    Given a policy returns estimates the state-action value function for this policy.
    
    Args:
        env: a MDP enviornment
        policy: a policy
        n_steps: number of iterations
        gamma: discount factor
        alpha: step size
    
    Returns:
        estimated state-action value function
    """
  
    # initialize the state-action value function and reset the environment at the initial sate
    Q = np.zeros((env.num_states, env.num_actions))
    env.reset()
    
    for n in range(n_steps):
        # sample state s_t and action a_t
        state = env.current_state
        action = np.random.choice(env.actions, p=policy[state])
        
        # given the action get the reward, the next state and if the state is terminal
        reward, next_state, done = env.step(action)
        
        # update the target (TD error)
        if done:
            target = reward
            env.reset()
        else:
            next_action = np.random.choice(env.actions, p=policy[next_state])
            target = reward + gamma * Q[next_state][next_action]
        
        # update the state-action value function
        Q[state][action] += alpha * (target - Q[state][action])
        
    return Q


def sarsa(env, n_steps, gamma, alpha, begin_epsilon, end_epsilon, verbose=False):
    
    """
    On policy algorithm
    Initializing an state-action value funtion it estimates the optimal state-action value function and the optimal policy.
    
    Args:
        env: a MDP enviornment
        n_steps: number of iterations
        gamma: discount factor
        begin_epsilon: initial exploration rate
        end_epsilon: minimum exploration rate
    
    Returns:
        optimal state-action value function
        optimal policy
    """
    
    
    # initialize the action-value function, a counter, the decay rule, the starting exploration rate
    Q = np.zeros((env.num_states, env.num_actions))
    
    t = 0
    decay_fn = linear_schedule(begin_epsilon, end_epsilon, 0, n_steps)
    epsilon = begin_epsilon

    # initialize the starting state and take an epsilon greedy action from that state
    state = env.reset()
    action = sample_action_eps_greedy_policy(env, state, Q, epsilon)

    
    for step in range(n_steps):

        # take the action and observe reward, next_state and if the state is terminal
        reward, next_state, done = env.step(action) 
        
        # take the next action from the next state (acting epsilon greedly)
        next_action = sample_action_eps_greedy_policy(env, next_state, Q, epsilon)
        
        # update the target (TD error) and the action-value function
        target = reward + gamma * Q[next_state, next_action]
        Q[state][action] += alpha * (target - Q[state][action])

        # update initial state and action
        state = next_state
        action = next_action

        # check if we are in terminal state
        if done:
            state = env.reset()
            action = sample_action_eps_greedy_policy(env, state, Q, epsilon)
        
        # update the exploration rate following the decay rule
        t += 1
        epsilon = decay_fn(t) # update epsilon

        if verbose and step % 500000 == 0:
            print(f'Passed {step} steps.')

    # compute optimal policy
    optimal_policy = compute_optimal_policy(env, Q)
  
    return optimal_policy, Q    

def q_learning(env, n_steps, gamma, alpha, begin_epsilon, end_epsilon, verbose=False):
    
    """
    Off policy algorithm
    Initializing an state-action value funtion it estimates the optimal state-action value function and the optimal policy.
    
    Args:
        env: a MDP enviornment
        n_steps: number of iterations
        gamma: discount factor
        begin_epsilon: initial exploration rate
        end_epsilon: minimum exploration rate
    
    Returns:
        optimal state-action value function
        optimal policy
    """
    
    # initialize the action-value function, a counter, the decay rule, the starting exploration rate
    Q = np.zeros((env.num_states, env.num_actions))
    t = 0
    decay_fn = linear_schedule(begin_epsilon, end_epsilon, 0, n_steps)
    epsilon = begin_epsilon
    
    # initialize the starting state
    state = env.reset()    
    
    for step in range(n_steps):
        # sample action following epsilon greedy policy
        action = sample_action_eps_greedy_policy(env, state, Q, epsilon)
        
        # take the action and observe reward, next_state and if the state is terminal
        reward, next_state, done = env.step(action)  
        
        # get the maximum q value among possible actions
        best_action_next = custom_argmax(Q[next_state], env.get_possible_actions(next_state))
        
        #update the target bootstrapping and the action-value function
        target = reward + gamma * Q[next_state][best_action_next]
        Q[state][action] += alpha * (target - Q[state][action])
        
        #update state
        state = next_state
        
        # check if the state is terminal
        if done:
            state = env.reset()
        # update the epsilon following the decay rule
        t += 1
        epsilon = decay_fn(t)        

        if verbose and step % 500000 == 0:
            print(f'Passed {step} steps.')
        
    # compute optimal policy
    optimal_policy = compute_optimal_policy(env, Q)

    return optimal_policy, Q