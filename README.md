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
A common failure mode for DDPG is that the learned Q-function begins to dramatically overestimate Q-values, which then leads to the policy breaking, because it exploits the errors in the Q-function.   
1. Clipped Double-Q Learning, see Deep Q-learning.   
2. Delayed Policy Updates, see Deep Q-learning.  
3. Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for the policy to exploit Q-function errors by smoothing out Q along changes in action.

## REINFROCE
[note](notebook/REINFORCE.ipynb), [code](algorithms/reinforce.py)    
Vanilla policy gradient.  
1. In practice, the computation of log probability can be inefficient. Notice that the log probability is an indicator of likelyhood. Thus, we can replace it with MSE o cross entropy loss. See [here](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf) for more details.     
2. The sum of reward term can have high variance, use baseline can help. Minus a constant from reward collected is unbiased in expectation, but it can reduce the variance. See [here](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf) for more details.  
3. The on-policy algorithm is not sample-efficient, importance sampling can help. See [here](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf) for more details.  

## PPO

## TRPO

## ACKER

## A3C  
Online actor-critic wih batches. 

## SAC