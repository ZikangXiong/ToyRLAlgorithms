from unittest import TestCase
import torch as th
from auto_rl.learning.network import MLP


class TestMLP(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMLP, self).__init__(*args, **kwargs)
        fc1 = th.nn.Linear(2, 3)
        bn1 = th.nn.BatchNorm1d(3)
        relu1 = th.nn.functional.relu
        fc2 = th.nn.Linear(3, 1)
        self.net = MLP([fc1, bn1, relu1, fc2])

    def test_forward(self):
        t = th.tensor([[1.0, 2.0], [1.0, 2.0]]).float()
        res = self.net(t)
        self.assertTrue(res.shape == th.Size([2, 1]), "shape error")

    def test_parameters_register(self):
        self.assertTrue(len([para for para in self.net.parameters()]) == 6, "parameter register error")
