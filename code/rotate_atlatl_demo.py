import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

from arm_robot import ArmRobot


def main():
    np.random.seed(42)
    arm_robot = ArmRobot()
    arm_robot.add_joint(
        "hip_joint",
        sphere_name="hip",
        box_name="upperbody",
        box_z=1.5,
        placement=pin.SE3(R.random().as_matrix(), np.array([0, 0, 0])),
    ).add_joint(
        "shoulder_joint",
        sphere_name="shoulder",
        box_name="upperarm",
        box_z=0.5,
        placement=pin.SE3(R.random().as_matrix(), np.array([0, 0, 1.5])),
    ).add_joint(
        "elbow_joint",
        sphere_name="elbow",
        box_name="lowerarm",
        box_z=0.5,
        placement=pin.SE3(R.random().as_matrix(), np.array([0, 0, 0.5])),
    ).add_joint(
        "wrist_joint",
        sphere_name="wrist",
        box_name="atlatl",
        box_z=0.75,
        placement=pin.SE3(R.random().as_matrix(), np.array([0, 0, 0.5])),
    ).add_joint(
        "P_joint",
        sphere_name="P",
        box_name="dart",
        box_z=0.75,
        placement=pin.SE3(R.random().as_matrix(), np.array([0, 0, 0.75])),
    )
    arm_robot.update_model()
    arm_robot.demo()


if __name__ == "__main__":
    main()
