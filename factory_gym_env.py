import gymnasium as gym
from gymnasium import spaces
import numpy as np
from factory_env import FactoryEnv
import pybullet as p

class FactoryGymEnv(gym.Env):
    def __init__(self, render=False):
        super(FactoryGymEnv, self).__init__()
        self.factory = FactoryEnv(render=render)
        
        # Actions: 
        # 0: Move to Conveyor (Pick position)
        # 1: Move to Assembly Station
        # 2: Move to Painting Station
        # 3: Move to Inspection Station
        # 4: Move to Output Station
        # 5: Pick/Place Action
        self.action_space = spaces.Discrete(6)
        
        # Observations:
        # Robot EE Pos (3), Nearest Door Pos (3), Door Status (1), Current Task (1)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(8,), dtype=np.float32)
        
        self.current_door = None
        self.steps = 0
        self.max_steps = 2000
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.factory.reset()
        self.current_door = self.factory.spawn_door()
        self.steps = 0
        return self._get_obs(), {}
        
    def _get_obs(self):
        ee_pos = self.factory.get_robot_ee_pos()
        door_pos, _ = p.getBasePositionAndOrientation(self.current_door["id"])
        
        # Just a simplified observation for now
        obs = np.array([
            ee_pos[0], ee_pos[1], ee_pos[2],
            door_pos[0], door_pos[1], door_pos[2],
            1.0 if self.current_door["status"] == "on_conveyor" else 0.0,
            len(self.current_door["tasks"]) / 3.0 # Progress
        ], dtype=np.float32)
        return obs
        
    def step(self, action):
        self.steps += 1
        reward = -0.01 # Step penalty
        done = False
        truncated = self.steps >= self.max_steps
        
        # Execute Action
        if action == 0: # Move to Conveyor
            target = [-0.3, -0.2, 0.75] # Approximated pick point
            self.factory.move_to(target)
        elif action == 1: # Assembly
            self.factory.move_to(self.factory.stations["assembly"]["pos"])
        elif action == 2: # Painting
            self.factory.move_to(self.factory.stations["painting"]["pos"])
        elif action == 3: # Inspection
            self.factory.move_to(self.factory.stations["inspection"]["pos"])
        elif action == 4: # Output
            self.factory.move_to(self.factory.stations["output"]["pos"])
        elif action == 5: # Pick/Place (Logic needed)
            reward += self._handle_pick_place()
            
        self.factory.step_simulation()
        
        obs = self._get_obs()
        
        # Reward for door moving through sequence
        if self.current_door["status"] == "completed":
            reward += 10.0
            done = True
            
        return obs, reward, done, truncated, {}

    def _handle_pick_place(self):
        # Implementation of simple pick/place logic
        # For now, just a placeholder and small reward for "trying"
        return 0.1

if __name__ == "__main__":
    env = FactoryGymEnv(render=True)
    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, trunc, _ = env.step(action)
        if done or trunc:
            obs, _ = env.reset()
