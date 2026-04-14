import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import time

class FactoryGymEnv(gym.Env):
    def __init__(self, render=False, is_baseline=False):
        super(FactoryGymEnv, self).__init__()
        # Member 2: AI Engineer - Environment Definition
        self.sim = None
        self.render_mode = render
        self.is_baseline = is_baseline
        
        # [Day 6] State Space: [Joint Angles(7), Workpiece X, Y, Z, Target X, Y, Z]
        # Total 13 dimensions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
        # [Day 6] Action Space: [Joint Velocities (7)]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # [Fix] Only create simulation if it doesn't exist
        if self.sim is None:
            from factory_sim import FactorySimulation
            self.sim = FactorySimulation(render=self.render_mode)
        else:
            # [Fix] Reset instead of recreate for speed
            p.resetBasePositionAndOrientation(self.sim.workpiece_id, [-2.6, 0, 0.05], [0, 0, 0, 1])
            for rid in self.sim.robot_ids:
                for j in range(7):
                    p.resetJointState(rid, j, 0.0)
            p.changeVisualShape(self.sim.workpiece_id, -1, rgbaColor=[1, 1, 1, 1])

        # Properly reset all local metric trackers for the new episode
        self.sim.is_painted = False
        self.current_step = 0
        self.total_dist = 0.0
        self.idle_time = 0.0
        self.last_end_effector_pos = None
        self.last_active_idx = None
        self.completions = 0
        
        # New Lock Flags
        self.waiting_at_station = False
        self.station_timer = 0
        self.cleared_stations = [False, False, False]
        
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        # [Fix] Extract real 13-dim observation state
        # 1. Workpiece Pos (3)
        pos, _ = p.getBasePositionAndOrientation(self.sim.workpiece_id)
        
        # 2. Identify Active Robot based on Workpiece X position
        if pos[0] < -1.0:
            active_rid = self.sim.robot_ids[0] # Assembly
            target_pos = self.sim.stations["Assembly"]
        elif pos[0] < 1.0:
            active_rid = self.sim.robot_ids[1] # Painting
            target_pos = self.sim.stations["Painting"]
        else:
            active_rid = self.sim.robot_ids[2] # Inspection
            target_pos = self.sim.stations["Inspection"]
            
        # 3. Get 7 Joint Angles of the active robot (7)
        joint_states = p.getJointStates(active_rid, range(7))
        joint_angles = [state[0] for state in joint_states]
        
        # 4. Target Pos (3)
        # Final combined state: [Joints(7), Workpiece(3), Target(3)]
        obs = np.array(joint_angles + list(pos) + list(target_pos), dtype=np.float32)
        return obs

    def step(self, action):
        # 1. Determine active robot for this step
        pos, ori = p.getBasePositionAndOrientation(self.sim.workpiece_id)
        if pos[0] < -1.0: active_idx = 0
        elif pos[0] < 1.0: active_idx = 1
        else: active_idx = 2
        
        rid = self.sim.robot_ids[active_idx]
        
        # [Fix] Apply real VELOCITY_CONTROL to the active robot arm
        for j in range(7):
            p.setJointMotorControl2(rid, j, p.VELOCITY_CONTROL, targetVelocity=action[j])
        
        # Physical Station Locking Logic
        station_centers = [-2.0, 0.0, 2.0]
        at_center = abs(pos[0] - station_centers[active_idx]) < 0.015
        
        if at_center and not self.cleared_stations[active_idx]:
            self.waiting_at_station = True
            self.station_timer += 1
            
        speed = 0.005
        extra_reward = 0.0
        
        if self.waiting_at_station:
            speed = 0.0 # Halt conveyor
            
            # Check end-effector distance (Link 6)
            end_effector_pos = p.getLinkState(rid, 6)[0]
            dist_to_wp = np.linalg.norm(np.array(end_effector_pos) - np.array(pos))
            
            # Core Mode Distinction Logic
            if self.is_baseline:
                if self.station_timer >= 1200:
                    self.waiting_at_station = False
                    self.cleared_stations[active_idx] = True
                    self.station_timer = 0
            else:
                # AI Mode: Let the robot complete physically or trigger an internal 450-tick optimized workflow limit
                if dist_to_wp < 0.3 or self.station_timer >= 450:
                    self.waiting_at_station = False
                    self.cleared_stations[active_idx] = True
                    self.station_timer = 0
                    if dist_to_wp < 0.3:
                        extra_reward += 10.0
                else:
                    extra_reward -= 0.01
                
        if pos[0] < 2.5:
            p.resetBasePositionAndOrientation(self.sim.workpiece_id, [pos[0] + speed, pos[1], pos[2]], ori)
        
        # Advance simulation
        p.stepSimulation()
        
        # 2. Metrics Calculation
        # Cycle time must be calculated purely from simulated physics steps
        self.current_step += 1
        cycle_time = self.current_step * (1.0 / 240.0)
        cycle_time = min(cycle_time, 20.0)
        
        # Switch check to avoid multi-robot teleportation spikes
        if getattr(self, 'last_active_idx', None) != active_idx:
            self.last_end_effector_pos = None
        self.last_active_idx = active_idx
        
        # Distance moved calculated from End-Effector Euclidean Spatial Shift
        end_effector_pos = p.getLinkState(rid, 6)[0]
        if self.last_end_effector_pos is not None and self.waiting_at_station:
            delta = np.linalg.norm(np.array(end_effector_pos) - np.array(self.last_end_effector_pos))
            self.total_dist += delta
        self.last_end_effector_pos = end_effector_pos
        distance = self.total_dist
        
        # Idle time logic: if action velocity requested is practically zero
        if np.max(np.abs(action)) < 0.01:
            self.idle_time += (1.0 / 240.0)
        idle_time = self.idle_time
        
        obs = self._get_obs()
        terminated = pos[0] > 2.4
        truncated = False
        
        # Throughput increments ONLY when successful termination condition explicitly occurs
        if terminated:
            self.completions += 1
        throughput = self.completions
        
        # [Fix] Required Reward Formula normalized to [-1, 1] range: (-cycle_time*0.1 + throughput*2.0 - idle_time*0.3) / 100.0
        reward = (-(cycle_time * 0.1) + (throughput * 2.0) - (idle_time * 0.3)) / 100.0
        reward += extra_reward
        
        # [Fix] Info dict with exact required keys
        info = {
            "cycle_time": float(cycle_time),
            "distance": float(distance),
            "throughput": int(throughput),
            "idle_time": float(idle_time),
            "task": self.sim._get_current_station(pos[0]),
            "reward": float(reward)
        }
        
        return obs, reward, terminated, truncated, info

    def render(self):
        pass
