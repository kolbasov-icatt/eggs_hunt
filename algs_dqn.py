from utils import *
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import random

from collections import deque 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Directory for saving models
save_dir = "trained_agents"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    

#Create a linear Neural Network Model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# Define the Agent class for the DQN and all the variables needed
class DQNAgent:
    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000) # add and remove from both ends is O(1), fixed size with automatic overflows, random access
        self.gamma = 0.9    # discount factor
        self.epsilon = 0.1   # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.998
        self.learning_rate = 0.0005
        self.checkpoint_episode = 50
        self.tau = 0.005
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.update_target_model()  # Initialize target model weights to match model weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        

    def update_target_model(self):
        """
        Update the target teta.
        """ 
        # Soft update of the target network's weights: θ′ ← τ θ + (1 −τ )θ′
        target_state_dict = self.target_model.state_dict()
        state_dict = self.model.state_dict()
        for key in state_dict:
            target_state_dict[key] = state_dict[key]*self.tau + target_state_dict[key]*(1-self.tau)
        self.target_model.load_state_dict(target_state_dict)
        
    def remember(self, state, action, reward, next_state, done):
        """
        Store the state, action, reward, next state and if the next state is terminal.
        """ 
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):       
        """
        Take a greedy action returning its index in the list of possible actions from that state.
        
        Args:
            state
        """ 
        state = torch.from_numpy(state).float().to(device)
        if np.random.rand() <= self.epsilon and training:
            return self.env.get_possible_actions(torch.IntTensor.item(state)).index( random.choice(self.env.get_possible_actions(torch.IntTensor.item(state))))
        act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()
            
    def replay(self, batch_size):     
        """
        Update the state-action value function and the loss, returning the final state-action value function.
        
        Args:
            batch_size: the size of a batch
        """ 
        
        # initialize a minibatch and a state-action value function
        minibatch = random.sample(self.memory, batch_size)
        Q=[]
        
        # store the state, the next state and the reward
        for state, action, reward, next_state, done in minibatch:
            state = torch.from_numpy(state).float().to(device)
            next_state = torch.from_numpy(next_state).float().to(device)
            reward = torch.tensor(reward).float().to(device)
            
            # check if the next state is final and update the target
            if done:
                target = reward
            else:
                target = reward + self.gamma * torch.max(self.target_model(next_state).detach()) # detach a tensor from the computation graph
            
            # evaluate the action-value function
            Q_sa = self.model(state)[0][action]
            Q.append(float(Q_sa))
            
            # calculate the loss function
            loss = self.criterion(target, Q_sa)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # update epsilon    
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return Q    
    
def DQN_train(env, batch_size, EPISODES, num_iterations):  
    """
    Train our DQN algorithm and output the state-action value.
    
    Args:
        env: a MDP enviornment
        batch_size: size of the batch
        EPISODES: number of episodes
        num_iterations: number of steps in each episode
    """ 
    
    # initialize the state, the state size, the action size, the agent and the return list
    state= env.reset()
    state_size = 1
    action_size = len(env.get_possible_actions(state))

    agent = DQNAgent(env,state_size, action_size)

    # in each episode reset the initial state and reshape it
    for e in range(1, EPISODES+1):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for t in range(num_iterations):
        
            # for each step in the episode take the action following the greedy policy
            action = agent.act(state) 

            # get the reward, the next state and if the next state is terminal
            reward,next_state,done = env.step(env.get_possible_actions(int(state))[action])
            
            # reshape the next state
            next_state = np.reshape(next_state, [1, state_size])

            # store state, action, reward, next_state and id the next state is terminal 
            agent.remember(state, action, reward, next_state, done) 

            # nupdate state
            state = next_state
            
            # check if it is terminal
            if done:
                print(f"episode: {e}/{EPISODES}, final step: {t}, epsilon: {agent.epsilon:.2f}")
                break
            
            # Get the state value function and the target updated
            if len(agent.memory) > batch_size:

                Q= agent.replay(batch_size) # time cost: 8 cs
                agent.update_target_model()

        # Save checkpoint every N episodes
        if e % agent.checkpoint_episode == 0:
            checkpoint_path = os.path.join(save_dir, f"model_checkpoint_episode_{e}.pth")
            torch.save(agent.model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint for episode {e} at '{checkpoint_path}'")
            
            
    return Q
