import sys
import time

import eigenpy
import numpy as np
import pinocchio as pin

from inverse_kinematics_solver import InverseKinematics
from robot import Robot

eigenpy.switchToNumpyArray()


class DropBall(Robot):
    def __init_for_demo__(self):
        self.initial_height = 2
        self.add_joint(
            "ball",
            geo_model=self.collision_model,
            joint_models=[pin.JointModelPZ()],
            placement=pin.SE3(np.eye(3), np.array([0, 0, self.initial_height])),
            inertias=[pin.Inertia.FromSphere(1, 0.1)],
        )
        self.ball = self.joints[0]
        self.collision_model.addCollisionPair(
            pin.CollisionPair(self.plane_geo_id, self.ball.geo_ids[0])
        )

    def demo(self, speed=1):
        self.displayCollisions(True)

        invk = InverseKinematics(self)
        invk.add_joint("ball", rotation=False)

        now = time.time()
        past = now
        total_time = 0

        while True:
            now = time.time()
            delta_time = now - past
            past = now
            total_time += delta_time

            invk.ball.translation[2] = (
                self.initial_height
                + ((1 / 2) * self.model.gravity.linear[2] * total_time**2) * speed
            )

            q_opt = invk.solve(invk.q)
            self.display(q_opt)
            pin.computeCollisions(self.collision_model, self.collision_data, True)
            if self.collision_data.collisionResults[0].isCollision():
                break

    def demo_integrate(self, speed=1):
        self.displayCollisions(True)

        a = self.model.gravity.linear[2:3]
        q = self.q0

        now = time.time()
        past = now
        total_time = 0

        while True:
            now = time.time()
            delta_time = now - past
            past = now
            total_time += delta_time

            q = pin.integrate(self.model, q, a * delta_time * speed)
            self.display(q)

            pin.computeCollisions(self.collision_model, self.collision_data, True)
            if self.collision_data.collisionResults[0].isCollision():
                break


if __name__ == "__main__":
    dropball = DropBall()
    dropball.init_for_demo()
    speed = 0.1
    if len(sys.argv) == 2 and sys.argv[1] == "c":
        with dropball.capture("dropball"):
            dropball.demo_integrate(speed)
            dropball.demo(speed)
    else:
        dropball.demo_integrate(speed)
        dropball.demo(speed)
