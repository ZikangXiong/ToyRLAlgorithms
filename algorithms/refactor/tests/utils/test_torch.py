from unittest import TestCase
import torch as th
from auto_rl.utils.torch import change_optim_lr, grad_clip, polyak_update


class Test(TestCase):
    def test_change_optim_lr(self):
        layer = th.nn.Linear(2, 2)
        opt = th.optim.Adam(params=layer.parameters(), lr=0.3)
        for param_group in opt.param_groups:
            self.assertTrue(param_group["lr"] != 0.1)
        change_optim_lr(opt, 0.1)
        for param_group in opt.param_groups:
            self.assertTrue(param_group["lr"] == 0.1)

    def test_grad_clip(self):
        layer = th.nn.Linear(2, 2)
        layer.weight.data = th.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=th.float32)
        loss = layer(th.tensor([1.0, 1.0], dtype=th.float32)).mean()
        loss.backward()

        self.assertTrue((layer.weight.grad == th.tensor([[0.5, 0.5], [0.5, 0.5]]).float()).all())
        grad_clip(layer, 0.1)
        self.assertTrue((layer.weight.grad == th.tensor([[0.1, 0.1], [0.1, 0.1]]).float()).all())

    def test_polyak_update(self):
        layer1 = th.nn.Linear(2, 1)
        layer1.weight.data = th.tensor([[1.0, 1.0]], dtype=th.float32)
        layer2 = th.nn.Linear(2, 1)
        layer2.weight.data = th.tensor([[2.0, 2.0]], dtype=th.float32)

        polyak_update(original_model=layer1, target_model=layer2, tau=0.5)
        self.assertTrue((layer2.weight == th.tensor([[1.5, 1.5]], dtype=th.float32)).all())
