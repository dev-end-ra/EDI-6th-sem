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
        
        # [Phase 5] Multi-Robot Assembly Line Setup
        self.robot_ids = []
        # Robot 1: Assembly (at -2)
        self.robot_ids.append(p.loadURDF("kuka_iiwa/model.urdf", [-2, 1.2, 0], useFixedBase=True))
        # Robot 2: Painting / Coloring (at 0)
        self.robot_ids.append(p.loadURDF("kuka_iiwa/model.urdf", [0, 1.2, 0], useFixedBase=True))
        # Robot 3: Inspection (at 2)
        self.robot_ids.append(p.loadURDF("kuka_iiwa/model.urdf", [2, 1.2, 0], useFixedBase=True))
        
        # [Day 4] Stations
        self.stations = {
            "Assembly": [-2, 0, 0.05],
            "Painting": [0, 0, 0.05],
            "Inspection": [2, 0, 0.05]
        }
        self._setup_stations()
        
        # [Phase 6] Car Model Integration
        self.workpiece_id = p.loadURDF("racecar/racecar.urdf", [-2.6, 0, 0.05], globalScaling=0.5)
        self.start_time = time.time()
        self.total_dist = 0.0
        self.last_robot_poses = [None] * 3
        self.is_painted = False

    def _setup_stations(self):
        for name, pos in self.stations.items():
            color = [0, 1, 0, 0.3] if name == "Assembly" else ([1, 0.5, 0, 0.3] if name == "Painting" else [0, 0.5, 1, 0.3])
            visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.01], rgbaColor=color)
            p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=pos)

    def step_simulation(self, manual_x=None, manual_data=None):
        """manual_data = {'arm_index': 0, 'joints': [...], 'x': ..., 'trigger_paint': bool}"""
        pos, ori = p.getBasePositionAndOrientation(self.workpiece_id)
        current_st = self._get_current_station(pos[0])
        
        # [Phase 6] Multi-Robot Logic & Manual Coloring
        for i, robot_id in enumerate(self.robot_ids):
            station_x = self.stations[list(self.stations.keys())[i]][0]
            dist_to_station = abs(pos[0] - station_x)
            
            if manual_data and manual_data.get('arm_index') == i:
                # Manual Control
                for j in range(7):
                    p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=manual_data['joints'][j])
            elif dist_to_station < 0.5:
                # Robot is active
                target_pos = [pos[0], pos[1], pos[2] + 0.15]
                
                # Special Coloring Motion for Robot 2
                if i == 1: # Painting Robot
                    # [Phase 6] Manual vs Auto Coloring
                    can_paint = True
                    if manual_data and manual_data.get('x') is not None:
                        # If in full manual mode, wait for trigger
                        can_paint = manual_data.get('trigger_paint', False)
                    
                    if can_paint or self.is_painted:
                        target_pos[1] += np.sin(time.time() * 10) * 0.3 # Sweep
                        # Visual Coloring (Industrial Orange-Gold)
                        p.changeVisualShape(self.workpiece_id, -1, rgbaColor=[1, 0.5, 0, 1])
                        self.is_painted = True
                    else:
                        target_pos = [pos[0], pos[1] + 0.4, pos[2] + 0.3] # Standby near car
                
                joint_poses = p.calculateInverseKinematics(robot_id, 6, target_pos)
                for j in range(min(7, len(joint_poses))):
                    p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=joint_poses[j])
            else:
                for j in range(7):
                    p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=0.0, positionGain=0.01)

        # Conveyor Logic
        if manual_data and manual_data.get('x') is not None:
            p.resetBasePositionAndOrientation(self.workpiece_id, [manual_data['x'], pos[1], pos[2]], ori)
        elif pos[0] < 2.5:
            speed = 0.005 if current_st == "Moving on Conveyor" else 0.001
            p.resetBasePositionAndOrientation(self.workpiece_id, [pos[0] + speed, pos[1], pos[2]], ori)
        else:
            # Continuous Production Reset
            p.resetBasePositionAndOrientation(self.workpiece_id, [-2.6, 0, 0.05], [0,0,0,1])
            p.changeVisualShape(self.workpiece_id, -1, rgbaColor=[1, 1, 1, 1]) 
            self.is_painted = False
            self.start_time = time.time()
            # Increment completion count in logger
            if hasattr(self, 'logger'):
                self.logger.completions += 1
            
        # Member 3: Distance & Efficiency Tracking (Phase 5: Multi-Arm)
        current_robot_poses = [p.getLinkState(rid, 6)[0] for rid in self.robot_ids]
        for i in range(3):
            if self.last_robot_poses[i]:
                self.total_dist += np.linalg.norm(np.array(current_robot_poses[i]) - np.array(self.last_robot_poses[i]))
        self.last_robot_poses = current_robot_poses

        cycle_time = time.time() - self.start_time
        self.logger.log_step(cycle_time, self.total_dist, 0.0, current_st)
        
        p.stepSimulation()
        time.sleep(1/240.)

    def _get_current_station(self, x):
        if -2.3 < x < -1.7: return "Assembly"
        if -0.3 < x < 0.3: return "Painting"
        if 1.7 < x < 2.3: return "Inspection"
        return "Moving on Conveyor"

if __name__ == "__main__":
    sim = FactorySimulation()
    print("🚀 Multi-Robot Car Coloring Line Active... Close window to stop.")
    while p.isConnected():
        sim.step_simulation()
