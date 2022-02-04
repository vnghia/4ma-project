import eigenpy
import numpy as np
import pinocchio as pin

from robot import Robot

eigenpy.switchToNumpyArray()


class LeggedRobot(Robot):
    def __init__(self, root_id=0, prefix=""):
        super().__init__(root_id, prefix)
        self.upperleg_len = 1
        self.lowerleg_len = 1
        self.leg_distance = 0.5
        self.foot_size = (0.4, 0.2, 0.1)
        self.foot_fore = 0.1
        self.body_size = (0.25, self.leg_distance, 0.1)
        self.leg_total_len = self.upperleg_len + self.lowerleg_len + self.foot_size[2]

        self.waist = None
        self.left_knee = None
        self.right_knee = None
        self.left_foot = None
        self.right_foot = None

    def __add_body(self):
        self.add_joint(
            "body",
            joint_models=[pin.JointModelPX(), pin.JointModelPY(), pin.JointModelPZ()],
            placement=pin.SE3(np.eye(3), np.array([0, 0, self.leg_total_len])),
            sphere_radius=0,
            box_x=self.body_size[0],
            box_y=self.body_size[1],
            box_z=self.body_size[2],
        )
        self.waist = self.joints[-1]

    def __add_leg(self, prefix, placement):
        self.add_joint(
            f"{prefix}_hip",
            parent=self.waist,
            joint_models=[pin.JointModelRX(), pin.JointModelRY(), pin.JointModelRZ()],
            sphere_name=f"{prefix}_hip",
            box_name=f"{prefix}_upperleg",
            box_z=self.upperleg_len,
            placement=placement,
            box_placement=pin.XYZQUATToSE3((0, 0, -self.upperleg_len / 2, 0, 0, 0, 1)),
        ).add_joint(
            f"{prefix}_knee",
            joint_models=[pin.JointModelRY()],
            sphere_name=f"{prefix}_knee",
            box_name=f"{prefix}_lowerleg",
            box_z=self.lowerleg_len,
            placement=pin.XYZQUATToSE3((0, 0, -self.upperleg_len, 0, 0, 0, 1)),
            box_placement=pin.XYZQUATToSE3((0, 0, -self.lowerleg_len / 2, 0, 0, 0, 1)),
        ).add_joint(
            f"{prefix}_ankle",
            joint_models=[pin.JointModelRY(), pin.JointModelRZ()],
            sphere_name=f"{prefix}_ankle",
            box_name=f"{prefix}_foot",
            box_x=self.foot_size[0],
            box_y=self.foot_size[1],
            box_z=self.foot_size[2],
            placement=pin.XYZQUATToSE3((0, 0, -self.lowerleg_len, 0, 0, 0, 1)),
            box_placement=pin.XYZQUATToSE3(
                (self.foot_fore, 0, -self.foot_size[2] / 2, 0, 0, 0, 1)
            ),
        )

    def __add_left_leg(self):
        self.__add_leg(
            "left",
            pin.XYZQUATToSE3((0, self.leg_distance / 2, 0, 0, 0, 0, 1)),
        )
        self.left_knee = self.joints[-2]
        self.left_foot = self.joints[-1]

    def __add_right_leg(self):
        self.__add_leg(
            "right",
            pin.XYZQUATToSE3((0, -self.leg_distance / 2, 0, 0, 0, 0, 1)),
        )
        self.right_knee = self.joints[-2]
        self.right_foot = self.joints[-1]

    def __init_for_demo__(self):
        self.__add_body()
        self.__add_left_leg()
        self.__add_right_leg()


if __name__ == "__main__":
    legged_robot = LeggedRobot()
    legged_robot.init_for_demo()
