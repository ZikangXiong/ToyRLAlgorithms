# Toy RL Algorithms

![code-grade](https://www.code-inspector.com/project/19281/status/svg)  
![logo](assets/logo.png)

A collection of pytorch implementation of basic reinforcement learning algorithms. This repository aims to provide
readable code and help understand the details and small ticks used in reinforcement learning algorithms. The
implementation is not suitable for any production environment or large scale experiments.

If you are looking for take-and-use code, these repositories will be helpful.

- [baselines](https://github.com/openai/baselines/): Openai RL algorithm implementation.
- [stable-baselines](https://github.com/hill-a/stable-baselines): More user-friendly implementation comparing with
  baselines.
- [stable-baselines zoo](https://github.com/araffin/rl-baselines-zoo): A collection of pretrained RL agents
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3): PyTorch implementation of stable-baselines
- [ray/rllib](https://github.com/ray-project/ray): Large scale training and turning.
- [ACME](https://github.com/deepmind/acme): Deep mind RL tools.

## Deep Q-learning
[note](notebook/deepQLearning.ipynb), [code](algorithms/dqn.py)  

Some tips are taken from UC Barkley CS 285. 
1. Replay buffer reduces the strong correlation between samples - [code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/2803522f2cbcf5cb2386eedf5b354016365ee5ad/algorithms/dqn.py#L81)  
2. Target network for solving the moving target - [code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/2803522f2cbcf5cb2386eedf5b354016365ee5ad/algorithms/dqn.py#L101)    
3. Ployak update for target network [code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/2803522f2cbcf5cb2386eedf5b354016365ee5ad/algorithms/dqn.py#L75)  
4. double q-learning reduce the over-fitting on noise - [code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/2803522f2cbcf5cb2386eedf5b354016365ee5ad/algorithms/dqn.py#L91)    
5. multi-step targets - [code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/2803522f2cbcf5cb2386eedf5b354016365ee5ad/algorithms/dqn.py#L90)    
6. Bellman error gradients can be big, use gradient clip and Huber loss - [code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/2803522f2cbcf5cb2386eedf5b354016365ee5ad/algorithms/dqn.py#L101-L108)  

## DDPG

[note](notebook/DDPG.ipynb), [code](algorithms/ddpg.py)   
Deep Q-learning with a neural argmaxer.  

## TD3

## REINFROCE
[note](notebook/REINFORCE.ipynb), [code](algorithms/reinforce.py)    
Vanilla policy gradient.  

## PPO

## TRPO

## ACKER

## A3C

## SAC