import numpy as np


class ReplayBuffer:

    def __init__(self, buffer_size, random_seed=0):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.buffer = []
        np.random.seed(random_seed)

    def add(self, s1, a, r, t, s2):
        experience = (s1, a, r, t, s2)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer = self.buffer[1:]
            self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        idx = np.random.randint(len(self.buffer), size=batch_size)
        batch = np.array(self.buffer, dtype=object)[idx]

        s1_batch = batch[:, 0]
        a_batch = batch[:, 1]
        r_batch = batch[:, 2]
        d_batch = batch[:, 3]
        s2_batch = batch[:, 4]

        def convert_type(obj_list):
            return [np.array(obj.tolist()) for obj in obj_list]

        return convert_type([s1_batch, a_batch, r_batch, d_batch, s2_batch])

    def clear(self):
        self.buffer.clear()


class OnPolicyBuffer:
    def __init__(self):
        self.buffer = [[], [], [], [], []]

    def add(self, s1, a1, r1, t, s2):
        self.buffer[0].append(s1)
        self.buffer[1].append(a1)
        self.buffer[2].append(r1)
        self.buffer[3].append(t)
        self.buffer[4].append(s2)

    def sample(self, batch_size):
        assert batch_size >= 0
        assert len(self.buffer[0]) > 0
        idx = np.random.randint(0, len(self.buffer[0]), batch_size)
        return (np.array(item)[idx] for item in self.buffer)

    def read(self):
        return (np.array(item) for item in self.buffer)

    def clear(self):
        self.buffer = [[], [], [], [], []]
