from unittest import TestCase
from auto_rl.learning.buffer import ReplayBuffer, OnPolicyBuffer

import numpy as np


def _fill_buffer(buffer, times):
    for i in range(times):
        s1 = np.array([i, i, i])
        a1 = np.array([i, i])
        r1 = i
        d1 = False
        s2 = np.array([i + 1, i + 1, i + 1])

        buffer.add(s1, a1, r1, d1, s2)


class TestReplayBuffer(TestCase):

    def test_add_excess_size(self):
        buffer = ReplayBuffer(5)
        _fill_buffer(buffer, 7)

        self.assertTrue((buffer.buffer[0][0] == np.array([2, 2, 2])).all(), "buffer size error")

    def test_size(self):
        buffer = ReplayBuffer(5)
        self.assertTrue(buffer.size() == 0)

        _fill_buffer(buffer, 1)
        self.assertTrue(buffer.size() == 1)

        _fill_buffer(buffer, 6)
        self.assertTrue(buffer.size() == 5)

    def test_sample_batch_number(self):
        buffer = ReplayBuffer(10)
        _fill_buffer(buffer, 10)

        samples = buffer.sample_batch(0)
        for s in samples:
            self.assertTrue(len(s) == 0)

        samples = buffer.sample_batch(2)
        for s in samples:
            self.assertTrue(len(s) == 2)

        samples = buffer.sample_batch(20)
        for s in samples:
            self.assertTrue(len(s) == 20)

    def test_sample_batch_type(self):
        buffer = ReplayBuffer(10)
        _fill_buffer(buffer, 10)
        samples = buffer.sample_batch(2)

        for s in samples:
            self.assertTrue(s.dtype != object)

    def test_random_seed(self):
        buffer1 = ReplayBuffer(100, random_seed=0)
        _fill_buffer(buffer1, 100)
        samples1 = buffer1.sample_batch(100)

        buffer2 = ReplayBuffer(100, random_seed=0)
        _fill_buffer(buffer2, 100)
        samples2 = buffer2.sample_batch(100)

        self.assertTrue((samples1[1] == samples2[1]).all())

    def test_clear(self):
        buffer = ReplayBuffer(100)
        _fill_buffer(buffer, 100)
        buffer.clear()
        self.assertTrue(buffer.buffer == [])


class TestOnPolicyBuffer(TestCase):
    def test_add(self):
        sample_num = 7
        buffer = OnPolicyBuffer()
        _fill_buffer(buffer, sample_num)

        for item in buffer.buffer:
            self.assertTrue(len(item) == sample_num, "buffer size error")

    def test_sample(self):
        sample_num = 7
        batch_size = 0
        buffer = OnPolicyBuffer()
        _fill_buffer(buffer, sample_num)

        samples = buffer.sample(batch_size)
        for item in samples:
            self.assertTrue(len(item) == batch_size)

        batch_size = 7
        buffer = OnPolicyBuffer()
        _fill_buffer(buffer, sample_num)

        samples = buffer.sample(batch_size)
        for item in samples:
            self.assertTrue(len(item) == batch_size)

        batch_size = 10
        buffer = OnPolicyBuffer()
        _fill_buffer(buffer, sample_num)

        samples = buffer.sample(batch_size)
        for item in samples:
            self.assertTrue(len(item) == batch_size)

    def test_read(self):
        sample_num = 7
        buffer = OnPolicyBuffer()
        _fill_buffer(buffer, sample_num)

        samples = buffer.read()
        for item in samples:
            self.assertTrue(len(item) == sample_num)

    def test_clear(self):
        sample_num = 7
        buffer = OnPolicyBuffer()
        _fill_buffer(buffer, sample_num)

        buffer.clear()
        self.assertTrue(buffer.buffer == [[], [], [], [], []])
