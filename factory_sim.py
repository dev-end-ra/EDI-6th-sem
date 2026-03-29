import pybullet as p
import pybullet_data
import time
import numpy as np

class FactorySimulation:
    def __init__(self, render=True, physics_speed=240.0):
        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Member 3: Phase 3 Data Pipeline
        from data_pipeline import FactoryLogger
        self.logger = FactoryLogger(mode="baseline")
        self.total_dist = 0.0
        self.idle_time = 0.0
        self.last_robot_pos = None
        
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
        self.workpiece_id = p.loadURDF("cube_small.urdf", [-2.6, 0, 0.1])
        self.start_time = time.time()
        
        # Member 3: Data Logging
        self.logs = []

    def _setup_stations(self):
        for name, pos in self.stations.items():
            color = [0, 1, 0, 0.3] if name == "Assembly" else ([1, 0.5, 0, 0.3] if name == "Painting" else [0, 0.5, 1, 0.3])
            visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.01], rgbaColor=color)
            p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=pos)

    def step_simulation(self, manual_x=None, manual_joint=None):
        pos, ori = p.getBasePositionAndOrientation(self.workpiece_id)
        
        # [Day 11] Manual Override Logic
        if manual_x is not None:
            new_pos = [manual_x, pos[1], pos[2]]
            p.resetBasePositionAndOrientation(self.workpiece_id, new_pos, ori)
        elif pos[0] < 2.5:
            # Baseline Automated Move
            current_station = self._get_current_station(pos[0])
            move_speed = 0.002 if current_station in ["Assembly", "Painting", "Inspection"] else 0.015
            new_pos = [pos[0] + move_speed, pos[1], pos[2]]
            p.resetBasePositionAndOrientation(self.workpiece_id, new_pos, ori)
            
        # [Day 11] Manual Robot Joint Override
        if manual_joint is not None:
            for j in range(7):
                # [Day 9] Smooth Motion: Use lower gains for manual control
                p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, 
                                        targetPosition=manual_joint[j],
                                        positionGain=0.03, velocityGain=1.0)
        else:
            # [Day 9] Smooth Wave
            joint_pos = np.sin(time.time() * 1.5) * 0.4
            p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, 
                                    targetPosition=joint_pos,
                                    positionGain=0.05)
        
        # Member 3: Distance Tracking
        robot_pos = p.getLinkState(self.robot_id, 6)[0] # End effector
        if self.last_robot_pos:
            self.total_dist += np.linalg.norm(np.array(robot_pos) - np.array(self.last_robot_pos))
        self.last_robot_pos = robot_pos

        # [Day 3/4/5] Core Logic
        status = self._get_current_station(pos[0])
        cycle_time = time.time() - self.start_time
        
        # Log to Phase 3 Pipeline
        self.logger.log_step(cycle_time, self.total_dist, self.idle_time, status)
        
        p.stepSimulation()
        time.sleep(1/240.)

    def _get_current_station(self, x):
        if -2.3 < x < -1.7: return "Assembly"
        if -0.3 < x < 0.3: return "Painting"
        if 1.7 < x < 2.3: return "Inspection"
        return "Moving on Conveyor"

    def get_logs_df(self):
        import pandas as pd
        return pd.DataFrame(self.logs).tail(20)

if __name__ == "__main__":
    sim = FactorySimulation()
    for _ in range(1000):
        sim.step_simulation()
    p.disconnect()
