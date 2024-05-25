from utils import *
import numpy as np
from tqdm import tqdm


def actor_critic_algorithm(env, n_steps, feature_extractor, alpha_w, alpha_theta, alpha_r, verbose=False):
    
    """
    For env with 2 rooms:
        w shape should be (1, 4)
        theta shape should be (1, 16)  16=4*4 where 4 is the number of actions
    """
    w = np.random.rand(feature_extractor.feature_dim_state()) * 0.01
    theta = np.random.rand(feature_extractor.feature_dim_state() * env.num_actions) * 0.01
    estimated_av_reward = 0
    
    # init state
    policy_param_theta = np.zeros((env.num_states, env.num_actions))
    for state in env.states:
        denominator = 0
        for action in env.get_possible_actions(state):
            x_state_action = feature_extractor.get_features_state_action(state, action)
            action_preference = x_state_action @ theta
            denominator += np.exp(action_preference)
        for action in env.actions:
            if action in env.get_possible_actions(state):
                x_state_action = feature_extractor.get_features_state_action(state, action)
                action_preference = x_state_action @ theta
                policy_param_theta[state][action] = np.exp(action_preference) / denominator
    
    state = env.reset()
    
    for step in tqdm(range(n_steps)):
        # STEP 1 #####################
        # take an action according to the policy parameterized by theta
        action = np.random.choice(env.actions, p=policy_param_theta[state])
        reward, next_state, done = env.step(action)
         
        # STEP 2. Compute the diffrential TD error #####################
        # features vector of a current state
        x_state = feature_extractor.get_features_state(state)  # [1, 0, 0, 0]
        x_next_state = feature_extractor.get_features_state(next_state) # [1, 0, 0.14, 0]
        
        # estimate state-value function
        est_V_state = x_state @ w
        est_V_next_state = x_next_state @ w
        
        td_error = reward - estimated_av_reward + est_V_next_state - est_V_state
        
        # STEP 3. update estimated average reward ###########################
        estimated_av_reward += alpha_r * td_error
        
        # STEP 4. Update the value function weights ##########################
        w += alpha_w * td_error * x_state
        
        # STEP 5. Update theta ACTOR ##########################
        # gradient of log pi (A|s; theta) 
        x_state_action = feature_extractor.get_features_state_action(state, action)  # (1x16)
        
        weighted_sum_over_actions = sum(
            policy_param_theta[state][action] * feature_extractor.get_features_state_action(state, action)
            for action in env.get_possible_actions(state)
        )
        
        gradient = x_state_action - weighted_sum_over_actions
        theta += alpha_theta * td_error * gradient
        
        # update policy
        for state in env.states:
            denominator = sum(
                np.exp(feature_extractor.get_features_state_action(state, action) @ theta)
                for action in env.get_possible_actions(state)
            )
            for action in env.actions:
                if action in env.get_possible_actions(state):
                    x_state_action = feature_extractor.get_features_state_action(state, action)
                    action_preference = x_state_action @ theta
                    policy_param_theta[state][action] = np.exp(action_preference) / denominator
        
        state = next_state
        
        if done:
            state = env.reset()
            if verbose:
                print('You reached the final step')
        
        if verbose and step==1:
            if td_error > 0:
                print(f'Step {step}. TD error ({td_error:.1f}) is positive.')
                print('It means, that the selected action resulted in a higher value than expected.')
                print('Take that action more often should improve the policy')
                print('So, the updated theta parameters will increase the probability of that action.')
            else:
                print(f'Step {step}. TD error ({td_error:.1f}) is negative.')
                print('It means, that the selected action resulted in a lower value than expected.')

    return policy_param_theta