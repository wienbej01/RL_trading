
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

class GymnasiumDummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_wait(self):
        results = [self.envs[i].step(self.actions[i]) for i in range(self.num_envs)]
        obs, rews, terminated, truncated, infos = zip(*results)
        return np.array(obs), np.array(rews), np.array(terminated), np.array(truncated), infos
