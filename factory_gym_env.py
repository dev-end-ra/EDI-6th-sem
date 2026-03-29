import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

class FactoryGymEnv(gym.Env):
    def __init__(self, render=False):
        super(FactoryGymEnv, self).__init__()
        # Member 2: AI Engineer - Environment Definition
        self.sim = None
        self.render_mode = render
        
        # [Day 6] State Space: [Joint Angles(7), Workpiece X, Y, Z, Target X, Y, Z]
        # Total 13 dimensions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
        # [Day 6] Action Space: [Joint Velocities (7)]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        from factory_sim import FactorySimulation
        self.sim = FactorySimulation(render=self.render_mode)
        
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        # Placeholder for real joint/pos extraction
        return np.zeros(13, dtype=np.float32)

    def step(self, action):
        # [Day 8] Refined Reward Function
        # 1. Action Penalty (Smoothness)
        reward = -0.01 * np.sum(np.square(action))
        
        # 2. Progress Reward (Moving workpiece along X)
        pos, _ = p.getBasePositionAndOrientation(self.sim.workpiece_id)
        reward += pos[0] * 0.1 # Encourage forward movement
        
        # 3. Precision Bonus (Station alignment)
        dist_to_station = 1.0 # Placeholder
        if dist_to_station < 0.1:
            reward += 5.0
            
        terminated = pos[0] > 2.4
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass
