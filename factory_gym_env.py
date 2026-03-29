import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

class FactoryGymEnv(gym.Env):
    def __init__(self, render=False):
        super(FactoryGymEnv, self).__init__()
        # Member 2: AI Engineer - Environment Definition
        
        # [Day 3] State Space: [Joint Angles(7), Workpiece X, Y, Z, Target X, Y, Z]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(13,), dtype=np.float32)
        
        # [Day 3] Action Space: [Joint Velocities (7)]
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        # reset sim logic here
        return np.zeros(13, dtype=np.float32), {}

    def step(self, action):
        # [Day 4] Draft Reward Function (Member 2)
        # Goal: Efficiently transport objects between stations
        
        # 1. Distance Penalty: Encourage robot to reach workpiece
        dist_to_obj = 0.5 # dummy
        reward = -dist_to_obj
        
        # 2. Cycle Time: Small penalty per step
        reward -= 0.01
        
        # 3. Precision: Bonus for hitting station center
        if dist_to_obj < 0.1:
            reward += 10
            
        return np.zeros(13, dtype=np.float32), reward, False, False, {}

    def render(self):
        pass
