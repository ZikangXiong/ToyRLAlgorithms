# Toy RL Algorithms

![code-grade](https://www.code-inspector.com/project/19281/status/svg)  
![logo](assets/logo.png)

A collection of pytorch implementation of basic reinforcement learning algorithms. This repository aims to provide readable code and help understand the details and small tricks used in reinforcement learning algorithms. The implementation is not suitable for any production environment or large scale experiments.

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
[spinning up docs](https://spinningup.openai.com/en/latest/algorithms/td3.html), [code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/master/algorithms/td3.py)  
A common failure mode for DDPG is that the learned Q-function begins to dramatically overestimate Q-values, which then leads to the policy breaking, because it exploits the errors in the Q-function.   
1. Clipped Double-Q Learning, see Deep Q-learning. - [code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/e714eaa9ae518d0be302ca54dcfe340a4991c817/algorithms/td3.py#L35)     
2. Delayed Policy Updates, see Deep Q-learning. - [code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/e714eaa9ae518d0be302ca54dcfe340a4991c817/algorithms/td3.py#L91)  
3. Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for the policy to exploit Q-function errors by smoothing out Q along changes in action. - [code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/e714eaa9ae518d0be302ca54dcfe340a4991c817/algorithms/td3.py#L40)  

## SAC
[paper](https://arxiv.org/abs/1801.01290), [spining up docs](https://spinningup.openai.com/en/latest/algorithms/sac.html), [code]()*  
Maximum entropy reinforcement learning.   
First, what’s similar to TD3?  
1. Like in TD3, both Q-functions are learned with MSE minimization, by regressing to a single shared target.  
2. Like in TD3, the shared target is computed using target Q-networks, and the target Q-networks are obtained by polyak averaging the Q-network parameters over the course of training.  
3. Like in TD3, the shared target makes use of the clipped double-Q trick.  

What’s different to TD3?    
1. Unlike in TD3, the target also includes a term that comes from SAC’s use of entropy regularization. Encourage exploration.   
2. Unlike in TD3, the next-state actions used in the target come from the current policy instead of a target policy.  
3. Unlike in TD3, there is no explicit target policy smoothing. TD3 trains a deterministic policy, and so it accomplishes smoothing by adding random noise to the next-state actions. SAC trains a stochastic policy, and so the noise from that stochasticity is sufficient to get a similar effect.  

## REINFROCE
[note](notebook/REINFORCE.ipynb), [code](algorithms/reinforce.py)    
Vanilla policy gradient.  
1. In practice, the computation of log probability can be inefficient. Notice that the log probability is an indicator of likelihood. Thus, we can replace it with MSE or cross entropy loss. See [here](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf) for more details.     
2. The sum of reward terms can have high variance, using a baseline can help. Minus a constant from reward collected is unbiased in expectation, but it can reduce the variance. See [here](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf) for more details.  
3. The on-policy algorithm is not sample-efficient, importance sampling can help. See [here](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf) for more details.  

## TRPO
[paper](https://arxiv.org/abs/1502.05477), [code]()*  
TRPO updates policies by taking the largest step possible to improve performance, while satisfying a special constraint on how close the new and old policies are allowed to be. The constraint is expressed in terms of KL-Divergence. 
1. Bounding the objective value [video](https://youtu.be/uR1Ubd2hAlE?list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&t=2403)  
2. Use KL-divergence, compute lambda in lagrange term. [video](https://youtu.be/uR1Ubd2hAlE?list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&t=2953)    
3. Linearize objective [video](https://youtu.be/uR1Ubd2hAlE?list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&t=3209)  
4. Natural gradient [video](https://youtu.be/uR1Ubd2hAlE?list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&t=3730)   

## ACKTR
[paper](https://arxiv.org/pdf/1708.05144.pdf), [code]()*  
Optimize both the actor and the critic using Kronecker-factored approximate
curvature (K-FAC) with trust region

## PPO
[paper](https://arxiv.org/pdf/1707.06347.pdf), [code](algorithms/ppo.py)  
Simple clip or lagrange multiplier* mimics the trust region policy optimization. - [code](https://github.com/ZikangXiong/ToyRLAlgorithms/blob/0952e0ad56eff4e5d98c155bd57b7f49071738ac/algorithms/ppo.py#L93)     
Generalized Advantage Estimation ([GAE](https://arxiv.org/pdf/1506.02438.pdf)) - [code](algorithms/utils/gae.py).   

## A3C  
[paper](https://arxiv.org/abs/1602.01783), [code]()*    
Online actor-critic with asynchronous gradient descent for optimization.
1. One can apply different exploration strategies to asynchronous environments, the authors claimed that will enhance the robustness.  

**\*: TODO**

