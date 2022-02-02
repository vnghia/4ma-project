import posixpath

import eigenpy
import numpy as np
import pinocchio as pin
from gepetto.color import Color

from display import Display

eigenpy.switchToNumpyArray()


class VisualPosition:
    # Class representing one 3D mesh of the robot, to be attached to a joint.
    # The class contains:
    # * name of the 3D objects inside Gepetto viewer.
    # * ID of the joint in the kinematic tree to which the body is attached.
    # * placement of the body with respect to the joint frame.

    # This class serves as an anchor for graphical elements.
    # As the underlying model processes, the rotation and translation
    # of this class will be reflected on the graphical element.

    # Note that no matter what the shape of `Visual`,
    # the undelying model contains only lines (body) and dots (joint).

    def __init__(self, name, placement):
        self.name = name
        self.placement = placement or pin.SE3.Identity()

    def calculate_new_pos(self, om_joint):
        return om_joint * self.placement

    def place(self, display, om_joint):
        om_joint = self.calculate_new_pos(om_joint)
        display.place(self.name, om_joint)


class Joint:
    def __init__(
        self, name, root_scene, model, gui, prefix="", parent=None, root_id=0, **kwargs
    ):
        self.name = name
        self.joint_models = kwargs.get("joint_models", [pin.JointModelRX()])
        self.placement = kwargs.get("placement", self.__calculate_placement(parent))
        self.ids = []
        self.inertias = kwargs.get("inertias")
        if not self.inertias:
            self.inertias = [None] * len(self.joint_models)
            for i in range(len(self.joint_models)):
                self.inertias[i] = pin.Inertia.Random()
        for i, (joint_model, inertia) in enumerate(
            zip(self.joint_models, self.inertias)
        ):
            # For the first joint model, we add it to the current parent. With other joint models,
            # we append it to the previous one, and place them in the same place.
            # So the last one will have all joint model combined and has the desired placement.
            current_parent = None
            current_placement = None
            if i == 0:
                current_parent = parent.id if parent else root_id
                current_placement = self.placement
            else:
                current_parent = self.ids[-1]
                current_placement = pin.SE3.Identity()
            self.ids.append(
                model.addJoint(
                    current_parent,
                    joint_model,
                    current_placement,
                    self.name,
                )
            )
            # Body is generally attached to the core of that joint,
            # so we want no rotation and no translation.
            model.appendBodyToJoint(self.ids[-1], inertia, pin.SE3.Identity())
        self.id = self.ids[-1]

        self.sphere_radius = kwargs.get("sphere_radius", 0.1)
        if self.sphere_radius:
            self.sphere_name = posixpath.join(
                root_scene, prefix, kwargs.get("sphere_name", f"sphere_{name}")
            )
            self.sphere_color = kwargs.get("sphere_color", Color.lightRed)
            gui.addSphere(self.sphere_name, self.sphere_radius, self.sphere_color)
            # For the joint, we want attach its anchor to its center of mass,
            # which is the center of the shpere, so we use `Identity` which means
            # no rotation and no translation.
            self.sphere_placement = kwargs.get("sphere_placement", pin.SE3.Identity())
            self.sphere = VisualPosition(self.sphere_name, self.sphere_placement)
        else:
            self.sphere = None

        self.box_x = kwargs.get("box_x", 0.1)
        self.box_y = kwargs.get("box_y", 0.1)
        self.box_z = kwargs.get("box_z", 1)
        if self.box_x and self.box_y and self.box_z:
            self.box_name = posixpath.join(
                root_scene, prefix, kwargs.get("box_name", f"box_{name}")
            )
            self.box_color = kwargs.get("box_color", Color.lightWhite)
            gui.addBox(
                self.box_name, self.box_x, self.box_y, self.box_z, self.box_color
            )

            # For the box, we want attach its controller the same position of the joint
            # as they have the same mouvement. However, by default, the `VisualPosition` is
            # attached to the center of mass which is in the middle of the box.
            # Here, we raise the box along its z-axis by `self.box_z / 2` to align its
            # `VisualPosition` with its lower end.
            # For more complex set the "box_placement".
            self.box_placement = kwargs.get(
                "box_placement", pin.SE3(np.eye(3), np.array([0, 0, self.box_z / 2]))
            )
            self.box = VisualPosition(self.box_name, self.box_placement)
        else:
            self.box = None

    def __calculate_placement(self, parent=None):
        if parent is None or parent.box_z == 0:
            return pin.SE3.Identity()
        else:
            return pin.SE3(np.eye(3), np.array([0, 0, parent.box_z]))


class Robot:
    # Define a class Robot.
    # The members of the class are:
    # * display: a display encapsulating a gepetto viewer client
    #   to create 3D objects and place them.
    # * model: the kinematic tree of the robot.
    # * data: the temporary variables to be used by the kinematic algorithms.
    # * visuals: the list of all the 'visual' 3D objects to render the robot,
    #   each element of the list being an object Visual (see above).

    def __init__(self, root_id=0, prefix=""):
        self.display = Display()
        self.visuals = []
        self.model = pin.Model()

        self.root_id = root_id
        self.prefix = prefix
        self.joints = []

        self.data = None
        self.q0 = None

    def add_joint(self, name, parent=None, **kwargs):
        parent = parent or (self.joints[-1] if self.joints else None)
        self.joints.append(
            Joint(
                name,
                self.display.root_scene,
                self.model,
                self.display.gui,
                self.prefix,
                parent,
                self.root_id,
                **kwargs,
            )
        )
        return self

    # Call this function after finish adding / updating joint to the model.
    def update_model(self, refresh_display=True):
        self.data = self.model.createData()
        self.q0 = np.zeros(self.model.nq)
        if refresh_display:
            # As we don't set position for the anchor when initializing Joint.
            # This is needed to place those anchors correctly.
            self.move_with_velocity(np.zeros(self.model.nq))

    def move_with_velocity(self, q, refresh_display=True):
        pin.forwardKinematics(self.model, self.data, q)
        names = []
        poss = []
        for joint in self.joints:
            omi = self.data.oMi[joint.id]
            if joint.sphere:
                names.append(joint.sphere_name)
                poss.append(pin.SE3ToXYZQUATtuple(joint.sphere.calculate_new_pos(omi)))
            if joint.box:
                names.append(joint.box_name)
                poss.append(pin.SE3ToXYZQUATtuple(joint.box.calculate_new_pos(omi)))
        self.display.places(names, poss)
        if refresh_display:
            self.display.refresh()

    def __init_for_demo__(self):
        self.add_joint("joint")

    def init_for_demo(self, refresh_display=True):
        if not self.q0 or not self.model:
            self.__init_for_demo__()
            self.update_model(refresh_display)

    def demo(self, dt=1e-3):
        pass


if __name__ == "__main__":
    robot = Robot()
    robot.init_for_demo()
    robot.demo()
