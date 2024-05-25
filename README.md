# RL Project: Eggs Hunting

In this project, we created a custom environment from scratch and implemented several Reinforcement Learning (RL) algorithms.

## Environment

Our environment allows for the creation of multiple rooms, each containing a configurable number of eggs. The probability of finding an egg in each room is adjustable and decreases with each successful find, making it progressively harder to locate the next egg. This decay factor is also customizable.

### Environment Features:
- **Customizable Room Count**: Set the number of rooms in the environment.
- **Adjustable Egg Count**: Place any number of eggs in each room.
- **Variable Egg-Finding Probability**: Define the probability of finding an egg in each room.
- **Egg-Finding Probability Decay**: Set the decay factor for egg-finding probability after each successful find in each room.

### Agent Actions:
- **Search**: The agent searches the current room for an egg.
  - Reward: +10 if an egg is found, -2 if not.
- **Stay**: The agent stays in the current room.
  - Reward: 0.
- **Go to Room X**: The agent moves to a different room.
  - Reward: -1.

## Algorithms

We implemented the following RL algorithms from scratch:
- **Dynamic Programming**
  - Policy Evaluation
  - Policy Improvement
  - Policy Iteration
  - Value Iteration
- **Monte Carlo Methods**
- **Temporal-Difference (TD) Learning**
- **SARSA**
- **Q-Learning**
- **Value Function Approximation (VFA)**
- **Deep Q-Network (DQN)**
- **Actor-Critic Methods**

## Training

The training process is detailed in the `training.ipynb` file, where we train our algorithms in the custom environment.

## Testing

We tested our algorithms with different numbers of rooms. The testing processes and results are documented in the `test.ipynb` and `test_actor_critic.ipynb` files.
