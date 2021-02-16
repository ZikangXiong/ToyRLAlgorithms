import numpy as np


class ReplayBuffer:

    def __init__(self, buffer_size, random_seed=0):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = []
        np.random.seed(random_seed)

    def add(self, s1, a, r, t, s2):
        experience = (s1, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer = self.buffer[1:]
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        idx = np.random.randint(len(self.buffer), size=batch_size)
        batch = np.array(self.buffer)[idx]

        s1_batch = batch[:, 0]
        a_batch = batch[:, 1]
        r_batch = batch[:, 2]
        d_batch = batch[:, 3]
        s2_batch = batch[:, 4]

        convert_type = lambda obj_list: [np.array(obj.tolist()) for obj in obj_list]
        return convert_type([s1_batch, a_batch, r_batch, d_batch, s2_batch])

    def clear(self):
        self.buffer.clear()
        self.count = 0
