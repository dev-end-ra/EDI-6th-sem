import pybullet as p
import numpy as np
import time

class RobotController:
    def __init__(self, robot_id, ee_index):
        self.robot_id = robot_id
        self.ee_index = ee_index
        self.grabbed_object = None
        self.constraint_id = None
        
    def move_to(self, target_pos, target_ori=None, duration=1.0):
        """Move transition with IK."""
        if target_ori is None:
            target_ori = p.getQuaternionFromEuler([0, 3.14, 0])
            
        joint_poses = p.calculateInverseKinematics(self.robot_id, self.ee_index, target_pos, target_ori)
        for i in range(len(joint_poses)):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, joint_poses[i])

    def pick(self, object_id):
        """Simulate a pick by creating a fixed constraint."""
        if self.grabbed_object is not None:
            return False
            
        # Get relative position and orientation
        ee_state = p.getLinkState(self.robot_id, self.ee_index)
        obj_pos, obj_ori = p.getBasePositionAndOrientation(object_id)
        
        # Check distance
        dist = np.linalg.norm(np.array(ee_state[0]) - np.array(obj_pos))
        if dist < 0.1: # Close enough to pick
            self.grabbed_object = object_id
            self.constraint_id = p.createConstraint(
                parentBodyCombinedIndex=self.robot_id,
                parentLinkIndex=self.ee_index,
                childBodyCombinedIndex=object_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0]
            )
            return True
        return False

    def place(self):
        """Release the grabbed object."""
        if self.grabbed_object is not None:
            p.removeConstraint(self.constraint_id)
            self.grabbed_object = None
            self.constraint_id = None
            return True
        return False

    def is_holding(self):
        return self.grabbed_object is not None
