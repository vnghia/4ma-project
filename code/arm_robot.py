import time

import eigenpy
import numpy as np
import pinocchio as pin

from robot import Robot

eigenpy.switchToNumpyArray()


class ArmRobot(Robot):
    def __init_for_demo__(self):
        for i in range(3):
            self.add_joint(str(i), box_z=1)

    def demo(self, dt=1e-3):
        for j in range(self.model.nv):
            v = np.array(self.model.nv * [0])
            v[j] = 1
            q = pin.neutral(self.model)
            for _ in range(1000):
                q += v * dt
                self.display(q)
                time.sleep(dt)
            for _ in range(1000):
                q -= v * dt
                self.display(q)
                time.sleep(dt)


if __name__ == "__main__":
    arm_robot = ArmRobot()
    arm_robot.init_for_demo()
    arm_robot.demo()
