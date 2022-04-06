import time

import eigenpy
import numpy as np
import pinocchio as pin
from scipy.interpolate import PchipInterpolator

from inverse_kinematics_solver import InverseKinematics
from robot import Robot

eigenpy.switchToNumpyArray()


class ConstantAtlatl(Robot):
    def __init__(self, *nargs, **kwargs):
        super().__init__(*nargs, **kwargs)
        self.__init_len_and_angles()
        self.invk = None

    def __init_len_and_angles(self):
        self.const_z = 1.5

        self.lens = [0.603, 0.286, 0.279, np.sqrt(0.067**2 + 0.2**2), 1]
        self.const_z = (
            self.lens[0]
            + self.lens[1] * np.cos(np.pi / 4)
            - self.lens[3] * np.sin(np.pi / 18)
        )

        self.angles = [np.pi / 24, np.pi / 24, np.pi / 24]
        self.angles.append(self.__calculate_last_angle(self.angles))
        self.angles.append(0)

        self.joint_names = ["hip", "shoulder", "elbow", "wrist", "P"]
        self.body_names = ["body", "upperarm", "lowerarm", "thrower", "atlatl"]

        self.joint_config = {
            "P": {
                "cylinder_radius": 0.025,
                "sphere_placement": pin.SE3(np.eye(3), np.array([0, -0.1, 0])),
            }
        }

    def __init_for_demo__(self):
        for joint_name, body_name, parent_len, angle, len in zip(
            self.joint_names,
            self.body_names,
            [0] + self.lens[:-1],
            self.angles,
            self.lens,
        ):
            placement = pin.XYZQUATToSE3(
                (
                    0,
                    (0 if joint_name != "P" else -0.1),
                    parent_len,
                    np.sin(angle / 2),
                    0,
                    0,
                    np.cos(angle / 2),
                )
            )
            self.add_joint(
                joint_name,
                placement=placement,
                cylinder_name=body_name,
                cylinder_z=len,
                **self.joint_config.get(joint_name, {})
            )
            setattr(self, joint_name, self.joints[-1])

    def __calculate_z(self, angles, lens):
        cumsum = np.cumsum(angles)
        ratio = np.cos(cumsum)
        return np.sum(lens * ratio), cumsum[-1]

    def __calculate_last_angle(self, *angles):
        current_z, current_angle = self.__calculate_z(angles, self.lens[:-2])
        required_z = self.const_z - current_z
        angle = 0
        if np.abs(required_z) < self.lens[-2]:
            angle = -np.arccos(required_z / self.lens[-2])
        return angle - current_angle

    def __construct_angle_process(self):
        process = np.empty((3, 3))
        const = self.const_z + self.lens[3] * np.sin(np.pi / 18)

        process[0] = [-np.pi / 6, 0, np.arccos((const - self.lens[1]) / self.lens[0])]
        process[1] = [
            -np.arccos((const - self.lens[0] * np.cos(np.pi / 6)) / self.lens[1]),
            -np.pi / 4,
            0,
        ]
        process[1] -= process[0]
        process[2] = [-np.pi / 2, -np.pi / 2, -np.pi / 2]
        process[2] -= process[1] + process[0]

        dshoulderhand = np.sqrt(
            (self.lens[1]) ** 2
            + self.lens[2] ** 2
            - 2 * np.cos(process[2, 2] + np.pi) * self.lens[1] * self.lens[2]
        )
        distance = self.const_z - self.lens[0] * np.cos(np.pi / 4)
        angleu = (self.lens[1] ** 2 + dshoulderhand**2 - self.lens[2] ** 2) / (
            2 * self.lens[1] * dshoulderhand
        )

        process_parent = np.empty((3, 4))
        process_parent[0] = [np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4]
        process_parent[1] = [
            angleu - np.pi / 2 + np.arcsin(distance / dshoulderhand),
            np.arccos(
                (distance**2 + self.lens[1] ** 2 - self.lens[2] ** 2)
                / (2 * self.lens[1] * distance)
            ),
            np.pi / 2,
            np.pi / 2,
        ]
        process_parent[1] -= process_parent[0]
        process_parent[2] = [
            process[2, 2],
            -np.arccos(
                (distance**2 + self.lens[2] ** 2 - self.lens[1] ** 2)
                / (2 * self.lens[2] * distance)
            )
            - np.pi / 4
            - process_parent[1, 1],
            -np.pi / 2,
            np.arccos((distance - self.lens[3] * np.cos(np.pi / 12)) / (self.lens[2]))
            - np.pi / 2,
        ]

        process = np.hstack([process, process_parent])
        return process

    def demo(self, speed=1):
        process = self.__construct_angle_process()
        times = np.arange(0, np.shape(process)[1]) * 1 / speed
        interp = [None] * np.shape(process)[0]

        for i, joint_process in enumerate(process):
            interp[i] = PchipInterpolator(times, joint_process)

        if not self.invk:
            invk = InverseKinematics(self, True)
            invk.add_joint("hip", translation=False)
            invk.add_joint("shoulder", translation=False)
            invk.add_joint("elbow", translation=False)
            invk.add_joint("wrist", translation=False)
            invk.add_joint("P", translation=False, relative=False)
            self.invk = invk

        now = time.time()
        past = now
        total_time = 0

        while total_time <= times[np.size(times) - 1]:
            now = time.time()
            delta_time = now - past
            past = now
            total_time += delta_time
            angles = np.empty(3)

            for i, (joint_process, joint_name) in enumerate(
                zip(process, self.joint_names[:-2])
            ):
                rot = interp[i](total_time)
                angles[i] = rot
                getattr(self.invk, joint_name).rotation = pin.Quaternion(
                    np.array(
                        [
                            np.sin(rot / 2),
                            0,
                            0,
                            np.cos(rot / 2),
                        ]
                    )
                ).toRotationMatrix()

            wrist_angle = self.__calculate_last_angle(angles)
            self.invk.wrist.rotation = pin.Quaternion(
                np.array(
                    [
                        np.sin(wrist_angle / 2),
                        0,
                        0,
                        np.cos(wrist_angle / 2),
                    ]
                )
            ).toRotationMatrix()

            self.invk.P.rotation = pin.Quaternion(
                np.array(
                    [
                        np.sin((5 * np.pi / 12) / 2),
                        0,
                        0,
                        np.cos((5 * np.pi / 12) / 2),
                    ]
                )
            ).toRotationMatrix()

            q_opt = self.invk.solve(self.invk.q)
            self.display(q_opt)

    def draw_line(self):
        try:
            from meshcat.geometry import LineSegments, PointsGeometry

            line = LineSegments(
                PointsGeometry(
                    position=np.array(
                        [
                            [0, -100, self.const_z],
                            [0, 100, self.const_z],
                        ]
                    )
                    .astype(np.float32)
                    .T
                )
            )
            self.viewer["line"].set_object(line)
        except Exception:
            return


if __name__ == "__main__":
    constant_atlatl = ConstantAtlatl()
    constant_atlatl.viz.clean()
    constant_atlatl.draw_line()
    constant_atlatl.init_for_demo()
    with constant_atlatl.capture():
        constant_atlatl.demo(5)
