import pybullet as p
import pybullet_data
import numpy as np
import time

class WarehouseSimulation:
    def __init__(self, render=True):
        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load ground
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Warehouse Dimensions
        self.width = 10
        self.height = 10
        
        # Shelves (Obstacles)
        self.shelves = []
        self._setup_warehouse()
        
        # AMR (Autonomous Mobile Robot)
        self.amr_id = self._spawn_amr([0, 0, 0.1])
        
        # Target Zone
        self.target_pos = [4, 4, 0.05]
        self._spawn_target(self.target_pos)

    def _setup_warehouse(self):
        # Create a grid of shelves
        for x in range(-4, 5, 2):
            for y in range(-4, 5, 2):
                if x == 0 and y == 0: continue # Leave home base clear
                shelf_id = p.loadURDF("cube_small.urdf", [x, y, 0.25], globalScaling=2)
                self.shelves.append(shelf_id)

    def _spawn_amr(self, pos):
        # Using a simple cube as AMR for now, can be replaced with a proper URDF
        amr_id = p.loadURDF("husky/husky.urdf", pos)
        return amr_id

    def _spawn_target(self, pos):
        # Visual indicator for target
        visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0, 1, 0, 1])
        p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=pos)

    def apply_action(self, linear_vel, angular_vel):
        """Apply velocity to Husky robot"""
        # Husky is a differential drive robot
        # We simplify to direct velocity control for RL
        left_v = linear_vel - angular_vel
        right_v = linear_vel + angular_vel
        
        # Husky joint indices (0, 1, 2, 3 - wheels)
        for i in range(4):
            p.setJointMotorControl2(self.amr_id, i, p.VELOCITY_CONTROL, targetVelocity=left_v if i%2==0 else right_v)

    def get_observation(self):
        pos, ori = p.getBasePositionAndOrientation(self.amr_id)
        # Simplified observation: [x, y, target_x, target_y]
        return np.array([pos[0], pos[1], self.target_pos[0], self.target_pos[1]], dtype=np.float32)

    def step(self):
        p.stepSimulation()
        time.sleep(1./240.)

    def disconnect(self):
        p.disconnect()

if __name__ == "__main__":
    sim = WarehouseSimulation()
    for _ in range(1000):
        sim.apply_action(0.5, 0.1)
        sim.step()
    sim.disconnect()
