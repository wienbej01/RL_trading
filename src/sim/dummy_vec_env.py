
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

class GymnasiumDummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)

    def reset(self):
        """Reset gymnasium-style envs and return only observations array.
        Discards the infos returned by env.reset()."""
        obs_list = []
        for env in self.envs:
            out = env.reset()
            if isinstance(out, tuple) and len(out) == 2:
                obs, _info = out
            else:
                obs = out
            obs_list.append(obs)
        return np.array(obs_list)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_wait(self):
        results = [self.envs[i].step(self.actions[i]) for i in range(self.num_envs)]
        obs, rews, terminated, truncated, infos = zip(*results)
        return np.array(obs), np.array(rews), np.array(terminated), np.array(truncated), infos
