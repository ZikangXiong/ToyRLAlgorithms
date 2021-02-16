from gym import Wrapper


class PixelReorderWrapper(Wrapper):

    def reset(self, **kwargs):
        return super(PixelReorderWrapper, self).reset(**kwargs).transpose((2, 0, 1))

    def step(self, action):
        obs, r, done, info = super(PixelReorderWrapper, self).step(action)
        obs = obs.transpose((2, 0, 1))

        return obs, r, done, info
