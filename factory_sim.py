import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import pandas as pd
from robot_agent import RoboticArmAgent
from task_manager import Task, TaskManager

class CarFactorySimulation:
    def __init__(self, num_robots=3):
        self.num_robots = num_robots
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load environment
        p.loadURDF("plane.urdf")
        
        # Workspace layout
        self.station_pos = [0, 0, 0.63]
        p.loadURDF("table/table.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
        
        # Robots and stations
        self.robots = []
        self.robot_home_positions = [
            [-0.35, -0.35, 0.63],
            [ 0.35, -0.35, 0.63],
            [ 0.0,   0.35, 0.63]
        ]
        
        for i in range(min(num_robots, len(self.robot_home_positions))):
            pos = self.robot_home_positions[i]
            robot_id = p.loadURDF("kuka_lwr/kuka.urdf", pos, [0, 0, 0, 1], useFixedBase=True)
            
            # Disable collisions with everything except the floor
            for link in range(-1, p.getNumJoints(robot_id)):
                p.setCollisionFilterGroupMask(robot_id, link, 0, 0)
                
            agent = RoboticArmAgent(robot_id, 6, [pos[0], pos[1], 0.9], name=f"Robot_{i+1}")
            self.robots.append(agent)
            
        # Car parts management
        self.car_parts = {}
        self.part_ids = {} # Task Name -> Body ID
        self.setup_car_parts()
        
        # Task Manager setup
        self.task_manager = TaskManager()
        self.setup_tasks()

    def setup_car_parts(self):
        """Pre-define the shapes for each assembly part."""
        # Chassis (Always there but starts semi-transparent)
        chassis_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.15, 0.04], rgbaColor=[0.6, 0.6, 0.6, 1])
        self.chassis_id = p.createMultiBody(baseVisualShapeIndex=chassis_v, basePosition=[0, 0, 0.75])
        
        # Frames (L/R)
        self.car_parts["Weld Left Frame"] = {
            "shape": p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.15, 0.08], rgbaColor=[0.4, 0.4, 0.7, 0]),
            "pos": [-0.23, 0, 0.85]
        }
        self.car_parts["Weld Right Frame"] = {
            "shape": p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.15, 0.08], rgbaColor=[0.4, 0.4, 0.7, 0]),
            "pos": [0.23, 0, 0.85]
        }
        
        # Axles
        self.car_parts["Front Axle"] = {
            "shape": p.createVisualShape(p.GEOM_CYLINDER, radius=0.015, length=0.35, rgbaColor=[0.2, 0.2, 0.2, 0]),
            "pos": [0, -0.15, 0.76], "ori": p.getQuaternionFromEuler([0, 1.57, 0])
        }
        self.car_parts["Rear Axle"] = {
            "shape": p.createVisualShape(p.GEOM_CYLINDER, radius=0.015, length=0.35, rgbaColor=[0.2, 0.2, 0.2, 0]),
            "pos": [0, 0.15, 0.76], "ori": p.getQuaternionFromEuler([0, 1.57, 0])
        }
        
        # Engine
        self.car_parts["Engine Install"] = {
            "shape": p.createVisualShape(p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.08], rgbaColor=[0.3, 0.3, 0.3, 0]),
            "pos": [0, -0.22, 0.85]
        }

    def setup_tasks(self):
        # Define assembly tasks
        t1 = Task("Chassis Base", 3, [0, 0, 0.9])
        t2 = Task("Weld Left Frame", 5, [-0.1, 0, 1.0], dependencies=["Chassis Base"])
        t3 = Task("Weld Right Frame", 5, [ 0.1, 0, 1.0], dependencies=["Chassis Base"])
        t4 = Task("Front Axle", 4, [0, -0.15, 0.95], dependencies=["Weld Left Frame", "Weld Right Frame"])
        t5 = Task("Rear Axle", 4, [0,  0.15, 0.95], dependencies=["Weld Left Frame", "Weld Right Frame"])
        t6 = Task("Engine Install", 6, [0, -0.2, 1.05], dependencies=["Front Axle"])
        t7 = Task("Painting", 7, [0, 0, 1.1], dependencies=["Engine Install", "Rear Axle"])
        
        for t in [t1, t2, t3, t4, t5, t6, t7]:
            self.task_manager.add_task(t)

    def run(self):
        print("Simulation started...")
        start_time = time.time()
        
        try:
            while not self.task_manager.is_all_completed():
                p.stepSimulation()
                
                # Update task manager and schedule
                self.task_manager.update()
                self.task_manager.schedule_tasks(self.robots)
                
                # Update robots
                for robot in self.robots:
                    robot.update(self.robots)
                
                # Real-time visualization of progress
                self.update_visuals()
                
                time.sleep(1./240.)
                
        except KeyboardInterrupt:
            p.disconnect()
            return
            
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nSimulation completed in {total_time:.2f} seconds.")
        print("-" * 30)
        print("Final Metrics:")
        total_energy = sum(r.energy_consumed for r in self.robots)
        print(f"Total Energy Consumption: {total_energy:.2f} units")
        
        for r in self.robots:
            print(f"[{r.name}] Energy: {r.energy_consumed:.2f}, Collisions avoided: {r.collisions_detected}")
            
        print("-" * 30)
        print("Tasks Workflow:")
        workflow_data = []
        for entry in self.task_manager.log:
            print(f"- {entry['task']} by {entry['robot']} at {entry['time']}")
            workflow_data.append([entry['task'], entry['robot'], entry['time']])
            
        # Save to CSV
        df_metrics = pd.DataFrame({
            "Metric": ["Total Time", "Total Energy"],
            "Value": [f"{total_time:.2f}s", f"{total_energy:.2f}"]
        })
        
        for r in self.robots:
            df_metrics = pd.concat([df_metrics, pd.DataFrame({
                "Metric": [f"{r.name} Energy", f"{r.name} Collisions"],
                "Value": [f"{r.energy_consumed:.2f}", r.collisions_detected]
            })], ignore_index=True)
            
        df_metrics.to_csv("simulation_metrics.csv", index=False)
        pd.DataFrame(workflow_data, columns=["Task", "Robot", "Time"]).to_csv("workflow_log.csv", index=False)
        print("\nResults saved to simulation_metrics.csv and workflow_log.csv")
        
        p.disconnect()

    def update_visuals(self):
        """Update part visibility and color based on completion."""
        for task_name in self.task_manager.completed_tasks:
            if task_name in self.car_parts and task_name not in self.part_ids:
                part = self.car_parts[task_name]
                # Spawn the part when completed
                color = [0.4, 0.4, 0.7, 1] if "Frame" in task_name else [0.2, 0.2, 0.2, 1]
                if task_name == "Engine Install": color = [0.3, 0.3, 0.3, 1]
                
                new_id = p.createMultiBody(
                    baseVisualShapeIndex=part["shape"], 
                    basePosition=part["pos"],
                    baseOrientation=part.get("ori", [0, 0, 0, 1])
                )
                p.changeVisualShape(new_id, -1, rgbaColor=color)
                self.part_ids[task_name] = new_id
        
        # Final Painting
        if "Painting" in self.task_manager.completed_tasks:
            shiny_color = [0.1, 0.1, 0.8, 1] # Metalli Blue
            p.changeVisualShape(self.chassis_id, -1, rgbaColor=shiny_color)
            for pid in self.part_ids.values():
                p.changeVisualShape(pid, -1, rgbaColor=shiny_color)

if __name__ == "__main__":
    sim = CarFactorySimulation(num_robots=3)
    sim.run()
