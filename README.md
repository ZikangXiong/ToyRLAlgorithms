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
5. multi-step target reward  - [code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/2803522f2cbcf5cb2386eedf5b354016365ee5ad/algorithms/dqn.py#L90)    
6. Bellman error gradients can be big, use gradient clip and Huber loss - [code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/2803522f2cbcf5cb2386eedf5b354016365ee5ad/algorithms/dqn.py#L101-L108)  

## DDPG

[note](notebook/DDPG.ipynb), [code](algorithms/ddpg.py)   
Deep Q-learning with a neural argmaxer.  

## TD3  
[code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/master/algorithms/td3.py)  
A common failure mode for DDPG is that the learned Q-function begins to dramatically overestimate Q-values, which then leads to the policy breaking, because it exploits the errors in the Q-function.   
1. Clipped Double-Q Learning, see Deep Q-learning. - [code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/e714eaa9ae518d0be302ca54dcfe340a4991c817/algorithms/td3.py#L35)     
2. Delayed Policy Updates, see Deep Q-learning. - [code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/e714eaa9ae518d0be302ca54dcfe340a4991c817/algorithms/td3.py#L91)  
3. Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for the policy to exploit Q-function errors by smoothing out Q along changes in action. - [code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/e714eaa9ae518d0be302ca54dcfe340a4991c817/algorithms/td3.py#L40)  

## SAC
[paper](https://arxiv.org/abs/1801.01290), [code]()  
Maximum entropy reinforcement learning.  

## REINFROCE
[note](notebook/REINFORCE.ipynb), [code](algorithms/reinforce.py)    
Vanilla policy gradient.  
1. In practice, the computation of log probability can be inefficient. Notice that the log probability is an indicator of likelyhood. Thus, we can replace it with MSE o cross entropy loss. See [here](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf) for more details.     
2. The sum of reward term can have high variance, use baseline can help. Minus a constant from reward collected is unbiased in expectation, but it can reduce the variance. See [here](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf) for more details.  
3. The on-policy algorithm is not sample-efficient, importance sampling can help. See [here](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf) for more details.  

## TRPO
TRPO updates policies by taking the largest step possible to improve performance, while satisfying a special constraint on how close the new and old policies are allowed to be. The constraint is expressed in terms of KL-Divergence. 
1. Bounding the objective value [video](https://youtu.be/uR1Ubd2hAlE?list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&t=2403)  
2. Use KL-divergence, compute lambda in lagrange term. [video](https://youtu.be/uR1Ubd2hAlE?list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&t=2953)    
3. Linearize objective [video](https://youtu.be/uR1Ubd2hAlE?list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&t=3209)  
4. Natural gradient [video](https://youtu.be/uR1Ubd2hAlE?list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&t=3730)   

## ACKTR
[paper](https://arxiv.org/pdf/1708.05144.pdf), [code]()  
Optimize both the actor and the critic using Kronecker-factored approximate
curvature (K-FAC) with trust region

## PPO
[paper](https://arxiv.org/pdf/1707.06347.pdf), [code]()  
Simple clip mimics the trust region policy optimization.   
Applied [GAE](https://arxiv.org/pdf/1506.02438.pdf) - [code](algorithms/utils/gae.py).   

## A3C  
[paper](https://arxiv.org/abs/1602.01783)  
Online actor-critic with synchronous gradient descent for optimization.