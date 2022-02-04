import eigenpy
import numpy as np
import pinocchio as pin
import scipy.optimize as scio

from callback import CallbackLogger

eigenpy.switchToNumpyMatrix()


class InverseKinematics:
    def __init__(self, robot) -> None:
        self.robot = robot
        self.q = pin.neutral(self.robot.model)
        self.joint_names = {}

    def add_joint(self, joint_name, translation=True, rotation=True):
        if joint_name not in self.joint_names:
            setattr(
                self,
                joint_name,
                self.robot.data.oMi[getattr(self.robot, joint_name).id].copy(),
            )
            self.joint_names[joint_name] = {
                "translation": translation,
                "rotation": rotation,
            }

    def cost(self, q):
        pin.forwardKinematics(self.robot.model, self.robot.data, q)

        error = 0

        for joint_name, joint_config in self.joint_names.items():
            cur = self.robot.data.oMi[getattr(self.robot, joint_name).id]
            ref = getattr(self, joint_name)
            if joint_config["translation"]:
                error += np.sum((cur.translation - ref.translation) ** 2)
            if joint_config["rotation"]:
                error += np.sum((cur.rotation - ref.rotation) ** 2)

        return error

    def solve(self, q):
        qopt = scio.fmin_bfgs(self.cost, q, callback=CallbackLogger(self), disp=False)
        return qopt
