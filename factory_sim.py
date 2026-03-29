import pybullet as p
import pybullet_data
import time
import numpy as np

class FactorySimulation:
    def __init__(self, render=True):
        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")
        
        # [Day 2] Conveyor Belt Logic
        # We simulate the belt as a visual area and move objects manually on it
        self.conveyor_pos = [0, 0, 0.01]
        self.conveyor_size = [5, 0.5, 0.02]
        self.belt_id = p.createVisualShape(p.GEOM_BOX, halfExtents=self.conveyor_size, rgbaColor=[0.2, 0.2, 0.2, 1])
        p.createMultiBody(baseVisualShapeIndex=self.belt_id, basePosition=self.conveyor_pos)
        
        # [Day 3] Robot Arm (Kuka IIWA as a substitute for UR5 for better stability)
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 1, 0], useFixedBase=True)
        
        # [Day 4] Stations
        self.stations = {
            "Assembly": [-2, 0, 0.05],
            "Painting": [0, 0, 0.05],
            "Inspection": [2, 0, 0.05]
        }
        self._setup_stations()
        
        # Active Workpiece
        self.workpiece_id = p.loadURDF("cube_small.urdf", [-2.4, 0, 0.1])
        
        # Member 3: Data Logging
        self.logs = []

    def _setup_stations(self):
        for name, pos in self.stations.items():
            color = [0, 1, 0, 0.3] if name == "Assembly" else ([1, 0.5, 0, 0.3] if name == "Painting" else [0, 0.5, 1, 0.3])
            visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.01], rgbaColor=color)
            p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=pos)

    def step_simulation(self):
        # Move workpiece on conveyor
        pos, ori = p.getBasePositionAndOrientation(self.workpiece_id)
        if pos[0] < 2.5:
            new_pos = [pos[0] + 0.01, pos[1], pos[2]]
            p.resetBasePositionAndOrientation(self.workpiece_id, new_pos, ori)
        
        # Simple Robot Wave (Day 3 Preview)
        joint_pos = np.sin(time.time() * 2) * 0.5
        p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=joint_pos)
        
        # Member 3: Log position
        self.logs.append({
            "timestamp": time.time(),
            "workpiece_x": round(pos[0], 3),
            "status": self._get_current_station(pos[0])
        })
        
        p.stepSimulation()
        time.sleep(1/240.)

    def _get_current_station(self, x):
        if -2.3 < x < -1.7: return "Assembly"
        if -0.3 < x < 0.3: return "Painting"
        if 1.7 < x < 2.3: return "Inspection"
        return "Conveyor"

    def get_logs_df(self):
        import pandas as pd
        return pd.DataFrame(self.logs).tail(20)

if __name__ == "__main__":
    sim = FactorySimulation()
    for _ in range(1000):
        sim.step_simulation()
    p.disconnect()
