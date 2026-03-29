import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulation_core import WarehouseSimulation
import pybullet as p

class WarehouseGymEnv(gym.Env):
    def __init__(self, render=False):
        super(WarehouseGymEnv, self).__init__()
        self.sim = WarehouseSimulation(render=render)
        
        # Action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # Observation space: [x, y, target_x, target_y, orientation_yaw]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(5,), dtype=np.float32)
        
        self.max_steps = 1000
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        self.sim = WarehouseSimulation(render=False) # Direct for reset, GUI handled at init
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        pos, ori = p.getBasePositionAndOrientation(self.sim.amr_id)
        euler = p.getEulerFromQuaternion(ori)
        # obs: [x, y, target_x, target_y, yaw]
        return np.array([pos[0], pos[1], self.sim.target_pos[0], self.sim.target_pos[1], euler[2]], dtype=np.float32)

    def step(self, action):
        self.sim.apply_action(action[0], action[1])
        self.sim.step()
        self.current_step += 1
        
        obs = self._get_obs()
        
        # Reward function
        dist = np.linalg.norm(obs[:2] - obs[2:4])
        reward = -dist # Penalize distance to target
        
        # Success bonus
        terminated = False
        if dist < 0.5:
            reward += 100
            terminated = True
        
        # Failure (out of bounds or timeout)
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
        
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        self.sim.disconnect()
