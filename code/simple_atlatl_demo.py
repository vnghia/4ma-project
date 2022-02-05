from arm_robot import ArmRobot


def main():
    arm_robot = ArmRobot()
    arm_robot.add_joint("hip", box_name="upperbody", box_z=1.5).add_joint(
        "shoulder", box_name="upperarm", box_z=0.5
    ).add_joint("elbow_joint", box_name="lowerarm", box_z=0.5).add_joint(
        "wrist", box_name="atlatl", box_z=0.75
    ).add_joint(
        "P", box_name="dart", box_z=0.75
    )
    arm_robot.rebuildData()
    arm_robot.demo()


if __name__ == "__main__":
    main()
