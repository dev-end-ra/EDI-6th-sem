import pybullet as p
import pybullet_data
import time
import numpy as np
import random

class FactoryEnv:
    def __init__(self, render=True):
        self.render = render
        if self.render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load environment
        p.loadURDF("plane.urdf")
        
        # Table/Factory Floor
        self.table_height = 0.6
        p.loadURDF("table/table.urdf", [0, 0, 0], [0, 0, 0, 1])
        
        # Conveyor Belt (Visual representation)
        # Positioned along the Y-axis (-0.5 to 0.5)
        conveyor_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.8, 0.01], rgbaColor=[0.2, 0.2, 0.2, 1])
        self.conveyor_id = p.createMultiBody(baseVisualShapeIndex=conveyor_v, basePosition=[-0.3, 0, self.table_height + 0.01])
        
        # Stations (Lined up on the other side of the table)
        # Assembly (Blue), Painting (Green), Inspection (Yellow), Output (Red)
        self.stations = {
            "assembly": {"pos": [0.3, -0.3, self.table_height + 0.02], "color": [0, 0, 1, 0.5]},
            "painting": {"pos": [0.3, -0.1, self.table_height + 0.02], "color": [0, 1, 0, 0.5]},
            "inspection": {"pos": [0.3, 0.1, self.table_height + 0.02], "color": [1, 1, 0, 0.5]},
            "output": {"pos": [0.3, 0.3, self.table_height + 0.02], "color": [1, 0, 0, 0.5]}
        }
        
        for name, data in self.stations.items():
            visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.08, 0.005], rgbaColor=data["color"])
            p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=data["pos"])
            
        # Robotic Arm (KUKA LWR)
        self.robot_pos = [0, 0, self.table_height]
        self.robot_id = p.loadURDF("kuka_lwr/kuka.urdf", self.robot_pos, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)
        self.ee_index = self.num_joints - 1
        
        # Door and Tasks
        self.doors = []
        self.current_door = None
        self.door_spawn_timer = 0
        self.conveyor_speed = 0.002
        
    def spawn_door(self):
        """Spawns a new door on the conveyor belt."""
        door_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.08, 0.12, 0.01], rgbaColor=[0.8, 0.8, 0.8, 1])
        door_c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.08, 0.12, 0.01])
        pos = [-0.3, -0.7, self.table_height + 0.05]
        door_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=door_c, baseVisualShapeIndex=door_v, basePosition=pos)
        
        door_info = {
            "id": door_id,
            "status": "on_conveyor", # on_conveyor, in_process, completed
            "tasks": ["assembly", "painting", "inspection"],
            "current_task": None
        }
        self.doors.append(door_info)
        return door_info

    def step_simulation(self):
        """Update simulation state."""
        p.stepSimulation()
        
        # Move doors on conveyor
        for door in self.doors:
            if door["status"] == "on_conveyor":
                pos, ori = p.getBasePositionAndOrientation(door["id"])
                new_pos = [pos[0], pos[1] + self.conveyor_speed, pos[2]]
                p.resetBasePositionAndOrientation(door["id"], new_pos, ori)
                
                # Recycle doors if they fall off or exit
                if new_pos[1] > 0.7:
                    p.removeBody(door["id"])
                    self.doors.remove(door)
                    
        # Periodic spawning
        self.door_spawn_timer += 1
        if self.door_spawn_timer > 500:
            self.spawn_door()
            self.door_spawn_timer = 0
            
    def get_robot_ee_pos(self):
        state = p.getLinkState(self.robot_id, self.ee_index)
        return np.array(state[0])

    def move_to(self, target_pos, target_ori=None):
        """Move robot end-effector to target position using IK."""
        if target_ori is None:
            target_ori = p.getQuaternionFromEuler([0, 3.14, 0]) # Pointing down
            
        joint_poses = p.calculateInverseKinematics(self.robot_id, self.ee_index, target_pos, target_ori)
        for i in range(len(joint_poses)):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, joint_poses[i])

    def reset(self):
        """Reset the environment."""
        p.resetSimulation()
        self.__init__(render=self.render)

if __name__ == "__main__":
    env = FactoryEnv(render=True)
    env.spawn_door()
    
    try:
        while True:
            env.step_simulation()
            # Basic test: Move robot to home
            env.move_to([0.2, 0, 0.9])
            time.sleep(1/240.)
    except KeyboardInterrupt:
        p.disconnect()
