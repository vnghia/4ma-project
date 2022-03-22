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


class ConstantAtlatl(Robot):
    def __init__(self, root_id=0, show_origin=True):
        super().__init__(root_id, show_origin)

        self.lens = {"body": 1, "upperarm": 0.6, "lowerarm": 0.4, "thrower": 0.3}
        self.names = {
            "hip": "body",
            "shoulder": "upperarm",
            "elbow": "lowerarm",
            "wrist": "thrower",
            "P": None,
        }

        self.constant_height = 2
        self.rotations = {"body": -np.pi / 6}
        self.placements = {
            name: pin.XYZQUATToSE3(
                (
                    0,
                    0,
                    0,
                    -np.sin(rotation / 2),
                    0,
                    0,
                    np.cos(rotation / 2),
                )
            )
            for name, rotation in self.rotations.items()
        }
        self.rotation_limits = {
            "hip": (-np.pi / 6, np.pi / 6),
            "shoulder": (0, np.pi / 2),
            "elbow": (-5 * np.pi / 6, 0),
            "wrist": (-np.pi / 2, np.pi / 2),
        }

        self.invk = None

    def __init_for_demo__(self):
        for joint_name, body_name in self.names.items():
            placement = self.placements.get(body_name)
            cylinder_z = self.lens.get(body_name, 0)
            self.add_joint(
                joint_name,
                placement=placement,
                cylinder_name=body_name,
                cylinder_z=cylinder_z,
            )
            setattr(self, joint_name, self.joints[-1])

    def draw_line(self):
        from meshcat.geometry import LineSegments, PointsGeometry

        l = LineSegments(
            PointsGeometry(
                position=np.array(
                    [
                        [0, -100, self.constant_height],
                        [0, 100, self.constant_height],
                    ]
                )
                .astype(np.float32)
                .T
            )
        )
        self.viewer["line"].set_object(l)

    def __constructing_rotation_process(self):
        total_time = 5
        process = np.empty((len(self.rotation_limits), total_time))
        for i, (lower, upper) in enumerate(self.rotation_limits.values()):
            process[i] = np.linspace(lower, upper, total_time)
        return process

    def demo(self):
        process = self.__constructing_rotation_process()
        times = np.arange(0, np.shape(process)[1])
        interp = [None] * np.shape(process)[0]

        joint_names = self.rotation_limits.keys()
        for i, (joint_process, joint_name) in enumerate(zip(process, joint_names)):
            interp[i] = PchipInterpolator(times, joint_process)

        if not self.invk:
            invk = InverseKinematics(self, True)
            invk.add_joint("hip", translation=False)
            invk.add_joint("shoulder")
            invk.add_joint("elbow")
            invk.add_joint("wrist")
            invk.add_joint("P")
            self.invk = invk
            self.invk.P.translation[2] = self.constant_height

        now = time.time()
        past = now
        total_time = 0

        while total_time <= times[np.size(times) - 1]:
            now = time.time()
            delta_time = now - past
            past = now
            total_time += delta_time

            for i, (joint_process, joint_name) in enumerate(zip(process, joint_names)):
                rot = interp[i](total_time)
                getattr(self.invk, joint_name).rotation = pin.Quaternion(
                    np.array(
                        [
                            -np.sin(rot / 2),
                            0,
                            0,
                            np.cos(rot / 2),
                        ]
                    )
                ).toRotationMatrix()

            q_opt = self.invk.solve(self.invk.q)
            self.display(q_opt)


if __name__ == "__main__":
    constant_atlatl = ConstantAtlatl()
    constant_atlatl.draw_line()
    constant_atlatl.init_for_demo()
    constant_atlatl.demo()
