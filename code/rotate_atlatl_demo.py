import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

from arm_robot import ArmRobot


def main():
    np.random.seed(42)
    arm_robot = ArmRobot()
    arm_robot.add_joint(
        "hip",
        placement=pin.SE3(R.random().as_matrix(), np.array([0, 0, 0])),
        box_name="upperbody",
        box_z=1.5,
    ).add_joint(
        "shoulder",
        placement=pin.SE3(R.random().as_matrix(), np.array([0, 0, 1.5])),
        box_name="upperarm",
        box_z=0.5,
    ).add_joint(
        "elbow",
        placement=pin.SE3(R.random().as_matrix(), np.array([0, 0, 0.5])),
        box_name="lowerarm",
        box_z=0.5,
    ).add_joint(
        "wrist",
        placement=pin.SE3(R.random().as_matrix(), np.array([0, 0, 0.5])),
        box_name="atlatl",
        box_z=0.75,
    ).add_joint(
        "P",
        placement=pin.SE3(R.random().as_matrix(), np.array([0, 0, 0.75])),
        box_name="dart",
        box_z=0.75,
    )
    arm_robot.rebuildData()
    arm_robot.demo()


if __name__ == "__main__":
    main()
