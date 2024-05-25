from utils import *
import numpy as np

def policy_evaluation(env, policy: np.array, gamma, threshold=0.00001, verbose=False):
    """
    Given a policy returns estimates the state-value function for this policy.
    
    Args:
        env: a MDP enviornment
        policy: a policy to be evaluated
        gamma: discount factor
        threshold: determines the accuracy of the evaluation
    
    Returns:
        estimated state-value function
    """
    
    # initialize the state value function and a counter
    count = 0
    V = np.zeros(env.num_states)
    
    while True:
        # initialize delta 
        delta = 0
        
        # take an initial state and check if it is terminal, if not continue 
        for state in env.states:
            if env.is_terminal_state(state):
                V[state] = 0
                continue
            
            # store the value of V
            old_value = V[state]
            new_value = 0
            
            # take an action from the ones that are possible in the state we are considering 
            for action in env.get_possible_actions(state):
                total_action = 0
                
                # update the value function
                for next_state, reward, prob in env.get_transition_info(state, action):
                    total_action += prob * (reward + gamma * V[next_state])                
                new_value += policy[state][action] * total_action
                #print(new_value)
            V[state] = new_value
            
            # update delta
            delta = max(delta, abs(old_value - new_value))
        
        count +=1 
        
        # rule to stop our execution 
        if delta < threshold:
            if verbose:
                print(f'State-value function is evaluated after {count} steps.')
                print(f'State-value function: {V}')
            break
    return V

def policy_improvement(env, V, gamma, verbose=False):
    
    """
    Given a state value function returns a policy and the state action value function.
    
    Args:
        env: a MDP enviornment
        V: a value function
        gamma: discount factor
        
    Returns:
        estimated state-action value function
        policy
    """
    
    #initialize the state action value funtion and a policy
    Q = np.zeros((env.num_states, env.num_actions))
    
    policy = np.zeros((env.num_states, env.num_actions))
    
    # Compute Q checking first if the state is not terminal and taking an action in the possible ones in that state
    for state in env.states:
        if env.is_terminal_state(state):
            for action in env.get_possible_actions(state):
                Q[state][action] = 0
        else:
            for action in env.get_possible_actions(state):
                total_return = 0
                for next_state, reward, prob in env.get_transition_info(state, action):
                    total_return += prob * (reward + gamma * V[next_state])
                Q[state][action] = total_return           
            
    # imporove policy in a greedy way 
    for state in env.states:
        if env.is_terminal_state(state):
            policy[state, 1] = 1  # in terminal state you should "stay"
        else:
            best_action = custom_argmax(Q[state], env.get_possible_actions(state))
            policy[state, best_action] = 1

    if verbose:
        print(f'State-action value function: {Q}')
    
    return policy, Q

def policy_iteration(env, gamma, threshold=0.0001, verbose=False):
    
    """
    Starting from a random policy it outputs the optimal state-value function , the optimal state-action value function and the optimal    policy.
    
    Args:
        env: a MDP enviornment
        gamma: discount factor
        threshold: determines the accuracy of the evaluation
        
    Returns:
        optimal state value function
        optimal state-action value function
        optimal policy
    """
    
    # initialize a random policy and a counter
    policy = create_random_policy(env)
    count = 0
    
    while True:
        is_policy_stable = True
        
        # Step 1: call policy evaluation 
        V = policy_evaluation(env, policy, gamma=gamma, threshold=threshold, verbose=verbose)

        # Step 2: call policy improvement
        new_policy, Q = policy_improvement(env, V, gamma=gamma, verbose=verbose)
        
        # Check for policy stability
        if not np.array_equal(new_policy, policy):
            is_policy_stable = False
        
        policy = new_policy
        count += 1

        if is_policy_stable:
            if verbose:
                print(f'Convergence achieved in {count} steps.')
            break
    
    return policy, V, Q

def value_iteration(env, gamma=0.9, threshold=0.00001, verbose=False):
    
    """
    Initializing the state value function, the state-action value function and a policy it output the optimal value function and the optimal policy.
    
    Args:
        env: a MDP enviornment
        gamma: discount factor
        threshold: determines the accuracy of the evaluation
        
    Returns:
        optimal state value function
        optimal policy
    """
    
    # initialize the state value function, the state action value function and a counter
    V = np.zeros(env.num_states)
    Q = np.zeros((env.num_states, env.num_actions))
    count = 0
    
    # Step 1: Find optimal state-value function V*
    while True:
        
        # initilize delta and check if the state is terminal
        delta = 0
        for state in env.states:
            if env.is_terminal_state(state):
                V[state] = 0
                continue
            # store the old value function, and taking an action from the possible ones update the state action value funtion, used for updating also the value funtion
            value_state_old = V[state]
            value = float('-inf')
            for action in env.get_possible_actions(state):
                total_return = 0
                for next_state, reward, prob in env.get_transition_info(state, action):
                    total_return += prob * (reward + gamma * V[next_state])
                Q[state][action] = total_return  
                
                if Q[state][action] > value:
                    value = Q[state][action]
            V[state] = value
            # update delta
            delta = max(delta, abs(value_state_old - V[state]) )
        
        # rule to stop the algorithm
        count += 1
        if delta < threshold:
            if verbose:
                print(f'Convergence achieved in {count} steps.')
            break
            
    # STep 2: Compute optimal policy
    #initialize a random policy
    policy = np.zeros((env.num_states, env.num_actions))
    
    # update the policy greedly
    for state in env.states:
        if env.is_terminal_state(state):
            policy[state, 1] = 1  # in terminal state you should "stay"
        else:
            best_action = custom_argmax(Q[state], env.get_possible_actions(state))
            policy[state, best_action] = 1
    
    return policy, V