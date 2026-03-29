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
    planeId = p.loadURDF("plane.urdf")
    
    # Load Table
    table_pos = [0, 0, 0]
    table_orientation = p.getQuaternionFromEuler([0, 0, 0])
    tableId = p.loadURDF("table/table.urdf", table_pos, table_orientation)
    
    # Load UR5 Robot
    # Note: Using the built-in Kuka model if UR5 is not directly available, 
    # but let's try to find a UR5-like model or use a generic robotic arm if needed.
    # PyBullet provides 'kuka_iiwa/model.urdf' by default.
    robot_pos = [0, 0, 0.63] # On top of the table
    robot_orientation = p.getQuaternionFromEuler([0, 0, 0])
    try:
        # Check if we can find a UR5 URDF or use Kuka as a baseline
        robotId = p.loadURDF("kuka_iiwa/model.urdf", robot_pos, robot_orientation, useFixedBase=True)
        print("Loaded Kuka IIWA robot model.")
    except Exception as e:
        print(f"Error loading robot model: {e}")
        return

    # Load Cube (object)
    cube_pos = [0.2, 0, 0.7]
    cube_orientation = p.getQuaternionFromEuler([0, 0, 0])
    cubeId = p.loadURDF("cube.urdf", cube_pos, cube_orientation, globalScaling=0.05)
    
    # Create Target Location (Visual only - a small sphere)
    target_pos = [0.4, 0.2, 0.7]
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
    target_id = p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=target_pos)
    
    print(f"Simulation setup complete. Robot ID: {robotId}, Table ID: {tableId}, Cube ID: {cubeId}, Target ID: {target_id}")
    
    # Add some debug sliders for joint control
    num_joints = p.getNumJoints(robotId)
    joint_indices = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robotId, i)
        if joint_info[2] == p.JOINT_REVOLUTE:
            p.addUserDebugParameter(f"Joint {i}", -math.pi, math.pi, 0)
            joint_indices.append(i)
            
    grab_button = p.addUserDebugParameter("Grab Cube", 1, 0, 0)
    release_button = p.addUserDebugParameter("Release Cube", 1, 0, 0)
    
    ee_link_index = 6
    constraint_id = -1
    
    # Read initial values to avoid auto-trigger at startup
    last_grab_click = p.readUserDebugParameter(grab_button)
    last_release_click = p.readUserDebugParameter(release_button)
    
    # Track joint slider values to avoid overriding the automatic sequence
    last_slider_values = [p.readUserDebugParameter(i) for i in range(len(joint_indices))]


    def move_to(target_pos):
        joint_poses = p.calculateInverseKinematics(robotId, ee_link_index, target_pos)
        for i, joint_idx in enumerate(joint_indices):
            # Using the first joint_poses values for the revolute joints
            p.setJointMotorControl2(robotId, joint_idx, p.POSITION_CONTROL, joint_poses[i])

    try:
        while True:
            p.stepSimulation()
            
            # Check buttons
            curr_grab_click = p.readUserDebugParameter(grab_button)
            if curr_grab_click > last_grab_click:
                last_grab_click = curr_grab_click
                print("Initiating Grab sequence...")
                
                # 1. Get cube position
                cube_pos, _ = p.getBasePositionAndOrientation(cubeId)
                
                # 2. Move to pre-pick (above cube)
                move_to([cube_pos[0], cube_pos[1], cube_pos[2] + 0.2])
                for _ in range(100): p.stepSimulation(); time.sleep(1./240.)
                
                # 3. Move to pick
                move_to([cube_pos[0], cube_pos[1], cube_pos[2] + 0.05])
                for _ in range(100): p.stepSimulation(); time.sleep(1./240.)
                
                # 4. Create constraint
                if constraint_id == -1:
                    constraint_id = p.createConstraint(robotId, ee_link_index, cubeId, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0.05])
                    print("Cube grabbed.")
                
                # 5. Lift up
                move_to([cube_pos[0], cube_pos[1], cube_pos[2] + 0.2])
                for _ in range(100): p.stepSimulation(); time.sleep(1./240.)

            curr_release_click = p.readUserDebugParameter(release_button)
            if curr_release_click > last_release_click:
                last_release_click = curr_release_click
                if constraint_id != -1:
                    p.removeConstraint(constraint_id)
                    constraint_id = -1
                    print("Cube released.")

            # Only update from sliders if the user actually moved them
            # This prevents the arm from snapping back after the grab sequence moves it
            for i, joint_idx in enumerate(joint_indices):
                target_pos_val = p.readUserDebugParameter(i)
                if abs(target_pos_val - last_slider_values[i]) > 0.001:
                    p.setJointMotorControl2(robotId, joint_idx, p.POSITION_CONTROL, target_pos_val)
                    last_slider_values[i] = target_pos_val
            
            time.sleep(1./240.)
    except KeyboardInterrupt:
        p.disconnect()
        print("Simulation stopped.")

if __name__ == "__main__":
    setup_simulation()
