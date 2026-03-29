import pybullet as p
import pybullet_data
import time
import math
import pandas as pd
import numpy as np

# Global variables for metrics
metrics_data = {
    "path_length": 0.0,
    "idle_time": 0.0,
    "last_ee_pos": None,
    "task_count": 0,
    "start_time": 0.0
}

def setup_simulation():
    # Connect to PyBullet GUI
    physicsClient = p.connect(p.GUI)
    
    # Set search path for built-in models
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Set gravity
    p.setGravity(0, 0, -9.81)
    
    # Load Plane
    p.loadURDF("plane.urdf")
    
    # Load Table
    table_pos = [0, 0, 0]
    p.loadURDF("table/table.urdf", table_pos, p.getQuaternionFromEuler([0, 0, 0]))
    
    # Load Kuka IIWA Robot
    robot_pos = [0, 0, 0.63] # On top of the table
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", robot_pos, [0, 0, 0, 1], useFixedBase=True)
    
    # Identify End Effector (usually the last link)
    ee_link_index = 6
    
    # Load Cube (object to pick)
    cube_pos = [0.2, 0, 0.72]
    cube_id = p.loadURDF("cube.urdf", cube_pos, [0, 0, 0, 1], globalScaling=0.05)
    
    # Define Target Location (Place position)
    target_pos = [0.4, 0.2, 0.72]
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=target_pos)
    
    print(f"Robot ID: {robot_id}, Cube ID: {cube_id}")

    def update_metrics():
        # Get EE state
        state = p.getLinkState(robot_id, ee_link_index, computeLinkVelocity=1)
        current_pos = state[0]
        linear_vel = state[6]
        
        # Path Length
        if metrics_data["last_ee_pos"] is not None:
            dist = math.sqrt(sum([(a-b)**2 for a, b in zip(current_pos, metrics_data["last_ee_pos"])]))
            metrics_data["path_length"] += dist
        metrics_data["last_ee_pos"] = current_pos
        
        # Idle Time (velocity < 0.01)
        vel_mag = math.sqrt(sum([v**2 for v in linear_vel]))
        if vel_mag < 0.01:
            metrics_data["idle_time"] += 1./240.

    def move_arm(target_pos):
        joint_poses = p.calculateInverseKinematics(robot_id, ee_link_index, target_pos)
        for i in range(len(joint_poses)):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i])
        
        # Wait a bit for the arm to move and collect metrics
        for _ in range(120):
            p.stepSimulation()
            update_metrics()
            time.sleep(1./240.)

    # Initialize Metrics
    metrics_data["start_time"] = time.time()
    metrics_data["last_ee_pos"] = p.getLinkState(robot_id, ee_link_index)[0]

    # Pick-and-Place Sequence
    time.sleep(1)
    
    print("Moving to pre-pick...")
    move_arm([0.2, 0, 0.9])
    
    print("Moving to pick...")
    move_arm([0.2, 0, 0.75])
    
    # "Pick"
    constraint_id = p.createConstraint(robot_id, ee_link_index, cube_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0.05])
    print("Picked cube.")
    
    time.sleep(0.5)
    move_arm([0.2, 0, 0.9])
    
    print("Moving to pre-place...")
    move_arm([0.4, 0.2, 0.9])
    
    print("Moving to place...")
    move_arm([0.4, 0.2, 0.75])
    
    # "Place"
    p.removeConstraint(constraint_id)
    metrics_data["task_count"] += 1
    print("Placed cube.")
    
    time.sleep(0.5)
    move_arm([0.4, 0.2, 0.9])
    
    print("Returning home...")
    move_arm([0, 0, 1.2])

    # Finalize Metrics
    end_time = time.time()
    cycle_time = end_time - metrics_data["start_time"]
    
    results = {
        "Metric": ["Cycle Time", "Path Length", "Idle Time", "Task Count"],
        "Value": [f"{cycle_time:.2f} sec", f"{metrics_data['path_length']:.2f} m", f"{metrics_data['idle_time']:.2f} sec", metrics_data["task_count"]]
    }
    
    df = pd.DataFrame(results)
    print("\n--- Workflow Metrics ---")
    print(df.to_string(index=False))
    
    csv_file = "metrics.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nResults saved to {csv_file}")

    try:
        # Keep simulation open for visual check
        for _ in range(240 * 2): # 2 more seconds
            p.stepSimulation()
            time.sleep(1./240.)
        p.disconnect()
        print("Simulation finished.")
    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    setup_simulation()
