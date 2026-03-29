import pybullet as p
import pybullet_data
import time
import numpy as np
import random
from robot_agent import RoboticArmAgent
from task_manager import Task, TaskManager

class CarFactorySimulation:
    def __init__(self, num_robots=3):
        self.num_robots = num_robots
        self.physicsClient = p.connect(p.DIRECT)
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
            [-0.5, -0.5, 0.63],
            [ 0.5, -0.5, 0.63],
            [ 0.0,  0.5, 0.63]
        ]
        
        for i in range(min(num_robots, len(self.robot_home_positions))):
            pos = self.robot_home_positions[i]
            robot_id = p.loadURDF("kuka_iiwa/model.urdf", pos, [0, 0, 0, 1], useFixedBase=True)
            agent = RoboticArmAgent(robot_id, 6, [pos[0], pos[1], 1.0], name=f"Robot_{i+1}")
            self.robots.append(agent)
            
        # Car components (Visual only for simulation)
        self.car_parts = {}
        self.parts_group = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.1, 0.05], rgbaColor=[0.5, 0.5, 0.5, 1])
        self.chassis_id = p.createMultiBody(baseVisualShapeIndex=self.parts_group, basePosition=[0, 0, 0.75])
        
        # Task Manager setup
        self.task_manager = TaskManager()
        self.setup_tasks()

    def setup_tasks(self):
        # Define assembly tasks
        t1 = Task("Chassis Base", 3, [0, 0, 0.82])
        t2 = Task("Weld Left Frame", 5, [-0.1, 0, 0.85], dependencies=["Chassis Base"])
        t3 = Task("Weld Right Frame", 5, [ 0.1, 0, 0.85], dependencies=["Chassis Base"])
        t4 = Task("Front Axle", 4, [0, -0.15, 0.8], dependencies=["Weld Left Frame", "Weld Right Frame"])
        t5 = Task("Rear Axle", 4, [0,  0.15, 0.8], dependencies=["Weld Left Frame", "Weld Right Frame"])
        t6 = Task("Engine Install", 6, [0, -0.2, 0.9], dependencies=["Front Axle"])
        t7 = Task("Painting", 7, [0, 0, 1.0], dependencies=["Engine Install", "Rear Axle"])
        
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
            
        print(f"\nSimulation completed in {total_time:.2f} seconds.")
        print("-" * 30)
        print("Final Metrics:")
        total_energy = sum(r.energy_consumed for r in self.robots)
        print(f"Total Energy Consumption: {total_energy:.2f} units")
        
        for r in self.robots:
            print(f"[{r.name}] Energy: {r.energy_consumed:.2f}, Collisions avoided: {r.collisions_detected}")
            
        print("-" * 30)
        print("Tasks Workflow:")
        for entry in self.task_manager.log:
            print(f"- {entry['task']} by {entry['robot']} at {entry['time']}")
            
        p.disconnect()

    def update_visuals(self):
        # Dynamically change car color based on progress
        completed = len(self.task_manager.completed_tasks)
        total = len(self.task_manager.tasks)
        progress = completed / total
        
        # Fade from grey to shiny blue as it builds
        color = [0.5 - 0.5 * progress, 0.5 + 0.3 * progress, 0.5 + 0.5 * progress, 1]
        p.changeVisualShape(self.chassis_id, -1, rgbaColor=color)
        
        # Overlay text indicating status (simple version)
        if completed < total:
            status_text = f"Assembly Status: {completed}/{total} tasks done"
            # In a real GUI, we could use p.addUserDebugText, but that can be messy
            # We'll just print status updates to the console for this simulation version.

if __name__ == "__main__":
    sim = CarFactorySimulation(num_robots=3)
    sim.run()
