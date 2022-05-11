import numpy as np


class Trajectory:
    def __init__(self):
        self.__init_len_and_angles()
        self.times = np.array([0, 1, 1, 1, 1, 1, 1])

    def __init_len_and_angles(self):
        self.const_z = 1.5

        self.lens = [0.603, 0.286, 0.279, np.sqrt(0.067**2 + 0.2**2), 1]
        self.const_z = (
            self.lens[0]
            + self.lens[1] * np.cos(np.pi / 4)
            - self.lens[3] * np.sin(np.pi / 18)
        )

        self.angles = [np.pi / 24, np.pi / 24, np.pi / 24]
        self.angles.append(self.calculate_last_angle(self.angles))
        self.angles.append(0)

        self.joint_names = ["hip", "shoulder", "elbow", "wrist", "P"]
        self.body_names = ["body", "upperarm", "lowerarm", "thrower", "atlatl"]

        self.joint_config = {
            "wrist": {"sphere_end_radius": 0.1},
            "P": {
                "cylinder_radius": 0.025,
                "sphere_radius": 0,
            },
        }

    def __calculate_z(self, angles, lens):
        cumsum = np.cumsum(angles)
        ratio = np.cos(cumsum)
        return np.sum(lens * ratio), cumsum[-1]

    def calculate_last_angle(self, *angles):
        current_z, current_angle = self.__calculate_z(angles, self.lens[:-2])
        required_z = self.const_z - current_z
        angle = 0
        if np.abs(required_z) < self.lens[-2]:
            angle = -np.arccos(required_z / self.lens[-2])
        return angle - current_angle

    def construct_angle_process(self):
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
