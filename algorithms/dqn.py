import os
import gym
import copy
import numpy as np
import torch as th
import torch.nn.functional as F

from algorithms.utils.networks import NatureCNN, MLP
from algorithms.utils.replay_buffer import ReplayBuffer
from algorithms.utils.wrappers import PixelReorderWrapper


class DQN:
    def __init__(self,
                 env: gym.Env,
                 pixel_input=False,
                 buffer_size=10000,
                 device="cpu",
                 seed=0
                 ):

        if pixel_input:
            self.env = PixelReorderWrapper(env)
            self.q_net = NatureCNN(env.observation_space.shape[0],
                                   env.observation_space.shape[1],
                                   env.action_space.n)
        else:
            self.env = env
            self.q_net = MLP(env.observation_space.shape[0],
                             env.action_space.n)

        self.target_net = copy.deepcopy(self.q_net)

        self.device = device
        self.q_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.eval()

        self.buffer = ReplayBuffer(buffer_size, random_seed=seed)
        self.env.seed(seed)
        th.manual_seed(seed)
        np.random.seed(seed)

        self.obs1 = None
        self.optimizer = th.optim.Adam(self.q_net.parameters(), lr=1e-3)

    def rollout(self, n_step, epsilon):
        if self.obs1 is None:
            self.obs1 = self.env.reset()
        reward_sum = 0

        for _ in range(n_step):
            if np.random.random_sample(1) > epsilon:
                action = self.predict(self.obs1)
            else:
                action = self.env.action_space.sample()

            obs2, reward, done, info = self.env.step(action)
            reward_sum += reward
            self.buffer.add(self.obs1, action, reward, done, obs2)
            self.obs1 = obs2

            if done:
                self.obs1 = self.env.reset()

        # simple check
        print(f"rollout cumulative reward: {reward_sum}")

    def update(self, n_epoch, training_step, batch_size, lr, gamma, tau, grad_clip):

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # polyak update target network after each epoch
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

        for ep in range(n_epoch):
            s1_batch, a_batch, r_batch, d_batch, s2_batch = self.buffer.sample_batch(batch_size)
            np2tensor = lambda tensor_list: [th.from_numpy(t).to(self.device) for t in tensor_list]
            s1_batch, a_batch, r_batch, d_batch, s2_batch = np2tensor([s1_batch, a_batch, r_batch, d_batch, s2_batch])

            # target q-value
            self.q_net.eval()
            with th.no_grad():
                # use q-net to predict action for double q-learning
                target_indices = self.q_net(s2_batch).detach().max(1)[1].unsqueeze(1)
                # TODO: consider sum up first t steps reward
                target_value = r_batch + gamma * self.target_net(s2_batch).detach(
                ).gather(1, target_indices).view(-1) * (~d_batch)
            self.q_net.train()

            for step in range(training_step):
                # predicted q-value
                a_batch = a_batch.long()
                pred_q_value = self.q_net(s1_batch).gather(1, a_batch.unsqueeze(1)).view(-1) * (~d_batch)

                # loss function, Huber loss
                loss = F.smooth_l1_loss(pred_q_value.float(), target_value.float())
                self.optimizer.zero_grad()
                loss.backward()

                # grad clip
                for param in self.q_net.parameters():
                    param.grad.data.clamp_(-grad_clip, grad_clip)
                self.optimizer.step()

        print(f"loss: {loss.detach().cpu().numpy().item()}")
        print(f"Target Q: {np.mean(target_value.detach().cpu().numpy())}")
        print(f"Predicted Q: {np.mean(pred_q_value.detach().cpu().numpy())}")
        print("---")

    def learn(self, time_step,
              n_step,
              n_epoch=3,
              training_step=10,
              batch_size=64,
              lr=1e-3,
              gamma=0.99,
              tau=0.99,
              grad_clip=1.0):

        for i in range(time_step // n_step):
            # epsilon-greedy, decay each rollout, for 50 rollouts.
            self.rollout(n_step, epsilon=max(0.5 - i * 0.01, 0))
            self.update(n_epoch, training_step, batch_size, lr, gamma, tau, grad_clip)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = th.from_numpy(obs).unsqueeze(0)
        self.q_net.eval()
        with th.no_grad():
            action = self.q_net(obs).numpy().reshape(-1).argmax()
        self.q_net.train()

        return action

    def save(self, file_path):
        save_dict = {
            "device": self.device,
            "pixel_input": True if type(self.q_net) is NatureCNN else False,
            "q_net_param": self.q_net.state_dict(),
            "target_net_param": self.q_net.state_dict()
        }

        if not os.path.isdir(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        th.save(save_dict, file_path)

    @classmethod
    def load(cls, file_path, env=None):
        save_dict = th.load(file_path)
        dqn = cls(env, pixel_input=save_dict["pixel_input"], device=save_dict["device"])
        dqn.q_net.load_state_dict(save_dict["q_net_param"])
        dqn.target_net.load_state_dict(save_dict["target_net_param"])

        return dqn
