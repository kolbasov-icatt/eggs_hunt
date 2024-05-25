from utils import *
import random

class EggsHuntEnv:
    def __init__(self, states, actions, transition_probabilities, total_eggs):
        
        self.state_names = states
        self.action_names = actions
        
        self.states = list(self.state_names.keys())
        self.actions = list(self.action_names.keys())
        
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)
        
        # the dynamics function        
        self.dynamics = transition_probabilities
        self.max_eggs = total_eggs
        self.current_state = None
        self.steps = 0

        
    def reset(self) -> int:
        """ Returns the initial state """
        self.current_state = self.states[0]
        self.steps = 0
        return self.current_state
        
            
    def step(self, action):
        """
        Make a step in the environment.

        Args:
            actions: the actions that takes an agent

        Returns:
            reward
            next_state
            done: True if it is the terminal state, False otherwise
        """
        
        # Rise some error if the environment doesn't work properly
        if self.current_state is None:
            raise RuntimeError('Current state is None. Call ".reset()" method')
        if self.is_done():
            raise RuntimeError('You reached the terminal state. Call ".reset()" method')
        if action not in self.get_possible_actions(self.current_state):
            raise ValueError(f"You can't do the action '{self.get_action_name(action)}' ({action}) in state '{self.get_state_name(self.current_state)}' ({self.current_state})")
        
        # store the dynamics
        possible_transitions = self.dynamics[self.current_state][action]
        
        # Unpack next states, rewards, and probabilities
        next_states, rewards, probabilities = zip(*possible_transitions)
        
        # Sample next state
        next_state = np.random.choice(next_states, p=probabilities)
        
        # Get reward
        reward = rewards[next_states.index(next_state)]
        
        # update state
        self.current_state = next_state
        self.steps += 1

        return reward, next_state, self.is_done()
    
    def get_possible_actions(self, state: int) -> List:
        """ 
        Returns possible actions in a state 
        
        """
        return list(self.dynamics[state].keys())
    
    def is_done(self) -> bool:
        """ 
        Returns True if the current state is a terminal state, False otherwise
        """
        return self.is_terminal_state(self.current_state)
    
    def is_terminal_state(self, state: int) -> bool:
        
        """ 
        Check if all the eggs have been collected
        Return true if they are, so the state is terminal, false otherwhise 
        
        """
        total_eggs = sum(self.get_state_name(state)[1:])
        return total_eggs == self.max_eggs
    
    def sample_action(self) -> int:
        """ Returns a random action from a current state """
        possible_actions = list(self.dynamics[self.current_state])
        random_action = random.choice(possible_actions)
        return random_action
    
    def get_transition_info(self, state, action):
        """ 
            Return: 
                reward, (transition_probability, next_state)
        """
        if action not in self.get_possible_actions(state):
            raise ValueError(f"You can't do the action '{self.get_action_name(action)}' ({action}) in state '{self.get_state_name(state)}' ({state})")
        
        possible_transitions = self.dynamics[state][action]
    
        return possible_transitions
        
    def get_state_name(self, state: int) -> str:
        " Returns the state name for a given state id "
        return self.state_names[state]
    
    def get_action_name(self, action: int) -> str:
        " Returns the action name for a given action id "
        return self.action_names[action]

