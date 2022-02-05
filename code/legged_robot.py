import itertools
import sys
import time

import eigenpy
import numpy as np
import pinocchio as pin
from scipy.interpolate import PchipInterpolator

from inverse_kinematics_solver import InverseKinematics
from robot import Robot

eigenpy.switchToNumpyArray()


class LeggedRobot(Robot):
    def __init__(self, root_id=0, show_origin=True):
        super().__init__(root_id, show_origin)
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

        self.invk = None

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
            placement=placement,
            box_name=f"{prefix}_upperleg",
            box_z=self.upperleg_len,
            box_placement=pin.XYZQUATToSE3((0, 0, -self.upperleg_len / 2, 0, 0, 0, 1)),
        ).add_joint(
            f"{prefix}_knee",
            joint_models=[pin.JointModelRY()],
            placement=pin.XYZQUATToSE3((0, 0, -self.upperleg_len, 0, 0, 0, 1)),
            box_name=f"{prefix}_lowerleg",
            box_z=self.lowerleg_len,
            box_placement=pin.XYZQUATToSE3((0, 0, -self.lowerleg_len / 2, 0, 0, 0, 1)),
        ).add_joint(
            f"{prefix}_ankle",
            joint_models=[pin.JointModelRY(), pin.JointModelRZ()],
            placement=pin.XYZQUATToSE3((0, 0, -self.lowerleg_len, 0, 0, 0, 1)),
            box_name=f"{prefix}_foot",
            box_x=self.foot_size[0],
            box_y=self.foot_size[1],
            box_z=self.foot_size[2],
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

    def __switch_foot__(self, process, number_step):
        index = np.empty(number_step, bool)
        index[::2] = False
        index[1::2] = True
        index = np.resize(index, (4, number_step)).T.ravel()
        process[1, index], process[2, index] = process[2, index], process[1, index]

    def __constructing_x_process(self, number_step, step_size=0.5):
        process = np.empty((3, number_step * 4 + 2))

        process[1, :-2] = np.repeat(np.arange(0, number_step), 4) * step_size
        process[1, -2:] = number_step * step_size

        process[2, 1:-1] = np.repeat(np.arange(1, number_step + 1), 4) * step_size
        process[2, 0] = 0
        process[2, -1] = number_step * step_size

        process[0, ::4] = process[1, ::4]
        process[0, 1::4] = process[0, ::4]
        process[0, 2::4] = (process[1, 2::4] + process[2, 2::4]) / 2
        process[0, 3::4] = process[2, 3::4]

        self.__switch_foot__(process[:, :-2], number_step)
        return process

    def __calculate_max_rise(self, step_size):
        return self.upperleg_len * (
            1 - np.cos(np.arcsin(step_size / self.upperleg_len))
        )

    def __calculate_min_waist(self, step_size):
        return (
            np.sqrt(self.upperleg_len**2 - (step_size / 2) ** 2) - self.upperleg_len
        )

    def __constructing_z_process(self, number_step, step_size=0.5):
        max_rise = self.__calculate_max_rise(step_size)
        min_waist = self.__calculate_min_waist(step_size)

        process = np.zeros((3, number_step * 4 + 2))

        process[0, 2::4] = min_waist
        process[1, 3::4] = max_rise
        process[2, 1:-1:4] = max_rise

        self.__switch_foot__(process[:, :-2], number_step)
        return process

    def move_step(self, number_step, step_size=0.5, speed=1):
        process = np.vstack(
            [
                self.__constructing_x_process(number_step, step_size),
                self.__constructing_z_process(number_step, step_size),
            ]
        )
        times = np.arange(0, np.shape(process)[1]) * 1 / speed
        interp = [None] * np.shape(process)[0]
        joint_names = ("waist", "left_foot", "right_foot")
        for i, (joint_process, joint_name) in enumerate(
            zip(process, itertools.cycle(joint_names))
        ):
            axis = 0 if i <= 2 else 2
            interp[i] = PchipInterpolator(
                times,
                joint_process
                + self.data.oMi[getattr(self, joint_name).id].translation[axis],
            )

        if not self.invk:
            invk = InverseKinematics(self)
            invk.add_joint("waist", rotation=False)
            invk.add_joint("left_knee", translation=False)
            invk.add_joint("right_knee", translation=False)
            invk.add_joint("left_foot", rotation=False)
            invk.add_joint("right_foot", rotation=False)
            self.invk = invk

        now = time.time()
        past = now
        total_time = 0

        while total_time <= times[np.size(times) - 2]:
            now = time.time()
            delta_time = now - past
            past = now
            total_time += delta_time

            for i, (joint_process, joint_name) in enumerate(
                zip(process, itertools.cycle(joint_names))
            ):
                axis = 0 if i <= 2 else 2
                getattr(self.invk, joint_name).translation[axis] = interp[i](total_time)

            q_opt = self.invk.solve(self.invk.q)
            self.display(q_opt)


if __name__ == "__main__":
    legged_robot = LeggedRobot()
    legged_robot.init_for_demo()
    if len(sys.argv) == 2 and sys.argv[1] == "c":
        with legged_robot.capture("legged_robot"):
            legged_robot.move_step(5, step_size=0.3, speed=7.5)
    else:
        legged_robot.move_step(5, step_size=0.3, speed=7.5)
