import pybullet as p
import numpy as np
import time
import random

class RoboticArmAgent:
    def __init__(self, robot_id, ee_link_index, home_pos, name="Robot"):
        self.robot_id = robot_id
        self.ee_link_index = ee_link_index
        self.home_pos = home_pos
        self.name = name
        
        self.status = "IDLE"  # IDLE, MOVING, WORKING, COMPLETED
        self.current_task = None
        self.target_pos = home_pos
        self.energy_consumed = 0
        self.collisions_detected = 0
        
        # Get joint information
        self.num_joints = p.getNumJoints(robot_id)
        self.revolute_joints = []
        for i in range(self.num_joints):
            info = p.getJointInfo(robot_id, i)
            if info[2] == p.JOINT_REVOLUTE:
                self.revolute_joints.append(i)

    def move_to(self, target_pos, threshold=0.01):
        """Set the target position for the arm."""
        self.target_pos = target_pos
        self.apply_control()

    def apply_control(self):
        """Calculate IK and apply position control to joints."""
        # Use rest poses to help IK convergence
        rest_poses = [0, 0, 0, 0, 0, 0, 0]
        joint_poses = p.calculateInverseKinematics(
            self.robot_id, 
            self.ee_link_index, 
            self.target_pos,
            lowerLimits=[-2.9, -2.0, -2.9, -2.0, -2.9, -2.0, -3.0],
            upperLimits=[ 2.9,  2.0,  2.9,  2.0,  2.9,  2.0,  3.0],
            jointRanges=[5.8, 4.0, 5.8, 4.0, 5.8, 4.0, 6.0],
            restPoses=rest_poses,
            maxNumIterations=500,
            residualThreshold=1e-5
        )
        
        for i in range(len(joint_poses)):
            p.setJointMotorControl2(
                self.robot_id, 
                i, 
                p.POSITION_CONTROL, 
                joint_poses[i],
                force=1000,
                maxVelocity=2.0
            )

    def is_at_target(self, threshold=0.1):
        """Check if the end effector is close to the target position."""
        state = p.getLinkState(self.robot_id, self.ee_link_index)
        curr_pos = np.array(state[0])
        dist = np.linalg.norm(curr_pos - np.array(self.target_pos))
        return dist < threshold

    def assign_task(self, task):
        """Assign a new task to the robot."""
        self.current_task = task
        self.status = "MOVING"
        print(f"[{self.name}] Assigned task: {task.name}")

    def check_collision(self, other_robots):
        """Check for proximity to other robots' end effectors."""
        state = p.getLinkState(self.robot_id, self.ee_link_index)
        curr_pos = np.array(state[0])
        
        for other in other_robots:
            if other.robot_id == self.robot_id:
                continue
            
            other_state = p.getLinkState(other.robot_id, other.ee_link_index)
            other_pos = np.array(other_state[0])
            
            dist = np.linalg.norm(curr_pos - other_pos)
            if dist < 0.25: # Safe distance
                return True
        return False

    def update(self, other_robots=[]):
        """Main update loop for the robot logic."""
        # Energy consumption update
        self.energy_consumed += self.get_energy_consumption() * (1./240.)

        if self.status == "MOVING":
            # Consistently apply control to move towards target
            self.apply_control()
            
            # Cooperative avoidance: wait if other robot is too close
            if self.check_collision(other_robots):
                self.collisions_detected += 1
                return
            
            state = p.getLinkState(self.robot_id, self.ee_link_index)
            curr_pos = np.array(state[0])
            dist = np.linalg.norm(curr_pos - np.array(self.target_pos))
            
            # Feedback for the user
            if random.random() < 0.01:
                print(f"[{self.name}] Moving... Dist to task: {dist:.3f}m")
            
            if self.is_at_target():
                if self.current_task:
                    self.status = "WORKING"
                    self.work_start_time = time.time()
                else:
                    self.status = "IDLE"
        
        elif self.status == "WORKING":
            # Simulate working time
            elapsed = time.time() - self.work_start_time
            if elapsed >= self.current_task.duration:
                print(f"[{self.name}] Completed task: {self.current_task.name}")
                self.current_task.complete()
                self.current_task = None
                self.status = "MOVING"
                self.target_pos = self.home_pos
                self.apply_control()
        
        elif self.status == "IDLE":
            # Just stay at home
            self.target_pos = self.home_pos
            self.apply_control()

    def get_energy_consumption(self):
        """Simplified energy model based on joint movement."""
        # In a real model, we'd look at motor torques, but here we can approximate
        # based on status and distance moved.
        if self.status == "WORKING":
            return 10.0 # Standard work power
        elif self.status == "MOVING":
            return 5.0  # Movement power
        return 1.0      # Idle power
