import pybullet as p
import pybullet_data
import time
import math

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

    def move_arm(target_pos):
        joint_poses = p.calculateInverseKinematics(robot_id, ee_link_index, target_pos)
        for i in range(len(joint_poses)):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i])
        
        # Wait a bit for the arm to move
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)

    # Pick-and-Place Sequence
    time.sleep(1)
    
    print("Moving to pre-pick...")
    move_arm([0.2, 0, 0.9])
    
    print("Moving to pick...")
    move_arm([0.2, 0, 0.75])
    
    # "Pick" - Create a constraint to attach cube to EE
    constraint_id = p.createConstraint(robot_id, ee_link_index, cube_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0.05])
    print("Picked cube.")
    
    time.sleep(0.5)
    move_arm([0.2, 0, 0.9])
    
    print("Moving to pre-place...")
    move_arm([0.4, 0.2, 0.9])
    
    print("Moving to place...")
    move_arm([0.4, 0.2, 0.75])
    
    # "Place" - Remove constraint
    p.removeConstraint(constraint_id)
    print("Placed cube.")
    
    time.sleep(0.5)
    move_arm([0.4, 0.2, 0.9])
    
    print("Returning home...")
    move_arm([0, 0, 1.2])

    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    setup_simulation()
