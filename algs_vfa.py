import numpy as np
import random
from utils import *

class FeatureExtractor:
    def __init__(self, env):
        self.env = env
        
    def feature_dim_state(self) -> int:
        return len(self.get_features_state(0))
    
    def get_features_state(self, state: int) -> np.array:
        state_name = self.env.get_state_name(state)
        room = state_name[0]  # 'Room 2' for example
        egg_counts = state_name[1:]
        n_rooms = len(state_name) - 1 # number of rooms in the env

        # One-hot encode the room information
        room_number = int(room.split()[1]) - 1
        one_hot_rooms = np.zeros(n_rooms)
        one_hot_rooms[room_number] = 1

        # Concatenate one-hot encoded room information with egg counts
        
        # normalize eggs_count
        egg_counts = [egg/self.env.max_eggs for egg in egg_counts]
        
        feature_vector = np.concatenate((one_hot_rooms, egg_counts))

        return feature_vector

    def get_features_state_action(self, state: int, action: int) -> np.array:
        feature_vector_state = self.get_features_state(state)
        
        feature_vector = []
        for a in self.env.actions:
            if a == action and a in self.env.get_possible_actions(state):
                feature_vector.append(feature_vector_state)
            else:
                  feature_vector.append(np.zeros(len(feature_vector_state)))        
        return np.concatenate(feature_vector)      

class VFA:
    def __init__(self, env, feature_extractor, alpha, gamma, epsilon):
        self.env = env
        self.feature_extractor = feature_extractor
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.weights = np.random.rand(feature_extractor.feature_dim_state())
        self.weights_sa = np.random.rand(feature_extractor.feature_dim_state() * self.env.num_actions)
        
    def gradient_MC_policy_evaluation(self, n_episodes, policy):
        x = self.feature_extractor
        for n in range(n_episodes):
            self.env.reset()

            # GENERATE AN EPISODE
            rewards = []
            choices = []
            while True:
                # choose an action following policy
                action = np.random.choice(self.env.actions, p=policy[self.env.current_state])
                choices.append((self.env.current_state, action))

                reward, next_state, done = self.env.step(action)
                rewards.append(reward)

                if done:
                    #print(f'Episode is finished in {len(choices)} steps')
                    break

            returns = compute_returns(rewards, gamma=self.gamma)
            for i, (state, action) in enumerate(choices):
                estimated_V = np.dot(self.weights, x.get_features_state(state))
                self.weights += self.alpha * (returns[i] -  estimated_V) * x.get_features_state(state)
            
        V_estimated = []
        for state in self.env.states:
            #print(x.get_features_state(state))
            V_estimated.append(self.weights @ x.get_features_state(state))

        return np.array(V_estimated)
    
    def semi_gradient_TD_policy_evaluation(self, n_steps, policy):
        x = self.feature_extractor
        state = self.env.reset()
        for n in range(n_steps):
            action = np.random.choice(self.env.actions, p=policy[state])
            reward, next_state, done = self.env.step(action)
            
            # update weights
            estimated_V = np.dot(self.weights, x.get_features_state(state))
            estimated_V_next = np.dot(self.weights, x.get_features_state(next_state))
            target = reward + self.gamma * estimated_V_next
            self.weights += self.alpha *  (target - estimated_V) * x.get_features_state(state)
            
            if done:
                #print(f'Terminal state is reached')
                state = self.env.reset()
            else:
                state = next_state
        
        V_estimated = []
        for state in self.env.states:
            #print(x.get_features_state(state))
            V_estimated.append(self.weights @ x.get_features_state(state))
            
        return np.array(V_estimated)
    
    def semi_gradient_sarsa(self, n_steps, begin_epsilon, end_epsilon):
        x = self.feature_extractor
        state = self.env.reset()
        
        # init
        t = 0
        decay_fn = linear_schedule(begin_epsilon, end_epsilon, 0, n_steps)
        epsilon = begin_epsilon       
        
        
        Q = np.zeros((self.env.num_states, self.env.num_actions))
        # select initial action
        action = sample_action_eps_greedy_policy(self.env, state, Q, epsilon)
        
        for n in range(n_steps):
            # take the action and observe reward, next_state
            reward, next_state, done = self.env.step(action)
            q_state_action = np.dot(self.weights_sa, x.get_features_state_action(state, action))

            if done:
                self.weights_sa += self.alpha * (reward - q_state_action) * x.get_features_state_action(state, action)
                state = self.env.reset()
                action = sample_action_eps_greedy_policy(self.env, state, Q, epsilon)
            else:
                # choose next action
                action_next = sample_action_eps_greedy_policy(self.env, state, Q, epsilon)

                # take the next action from the next state (acting epsilon greedly)
                next_action = sample_action_eps_greedy_policy(self.env, next_state, Q, epsilon)
                target = reward + self.gamma * np.dot(self.weights_sa, x.get_features_state_action(next_state, next_action))
                self.weights_sa += self.alpha * (target - q_state_action) * x.get_features_state_action(state, action)
            
                # update initial state and action
                state = next_state
                action = next_action
            
            # update the exploration rate following the decay rule
            t += 1
            epsilon = decay_fn(t) # update epsilon
            
        Q_estimated = []        
        for state in self.env.states:
            q_state = [ self.weights_sa @ x.get_features_state_action(state, action) for action in self.env.actions]
            Q_estimated.append(q_state)
        return np.array(Q_estimated)
    
    def semi_gradient_qlearning(self, n_steps, begin_epsilon, end_epsilon):
        x = self.feature_extractor
        state = self.env.reset()
        
        # init
        t = 0
        decay_fn = linear_schedule(begin_epsilon, end_epsilon, 0, n_steps)
        epsilon = begin_epsilon       
        
        
        Q = np.zeros((self.env.num_states, self.env.num_actions))
        # select initial action
        
        for n in range(n_steps):
            action = sample_action_eps_greedy_policy(self.env, state, Q, epsilon)

            # take the action and observe reward, next_state
            reward, next_state, done = self.env.step(action)
            
            # get the maximum q estimated value among possible actions
            maximum = float('-inf')
            for next_action in self.env.get_possible_actions(next_state):
                est_q = x.get_features_state_action(next_state, next_action) @ self.weights_sa
                if est_q > maximum:
                    maximum = est_q
            
            # update the weights
            q_estimated = x.get_features_state_action(state, action) @ self.weights_sa
            self.weights_sa += self.alpha * (reward + self.gamma * maximum - q_estimated) * x.get_features_state_action(state, action) 
                
            state = next_state

            if done:
                state = self.env.reset()
            
            # update the exploration rate following the decay rule
            t += 1
            epsilon = decay_fn(t) # update epsilon
            
        Q_estimated = []        
        for state in self.env.states:
            q_state = [ self.weights_sa @ x.get_features_state_action(state, action) for action in self.env.actions]
            Q_estimated.append(q_state)
        return np.array(Q_estimated)
   
