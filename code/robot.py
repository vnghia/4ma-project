import time
from contextlib import contextmanager

import eigenpy
import hppfcl
import numpy as np
import pinocchio as pin

eigenpy.switchToNumpyArray()

RED = np.array([1, 0, 0, 1])
WHITE = np.array([0.85, 0.85, 0.85, 0.85])


class Joint:
    def __init__(self, name, model, geo_model, parent=None, root_id=0, **kwargs):
        self.name = name
        self.joint_models = kwargs.get("joint_models", [pin.JointModelRX()])
        self.placement = kwargs.get("placement", self.__calculate_placement(parent))
        self.ids = []
        self.geo_ids = []
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
        self.sphere = None
        if self.sphere_radius:
            self.sphere_name = kwargs.get("sphere_name", self.name)
            self.sphere_color = kwargs.get("sphere_color", RED)
            # Placement of the geometry with respect to the joint frame
            self.sphere_placement = kwargs.get("sphere_placement", pin.SE3.Identity())
            self.sphere = self.__create_geo_obj(
                self.sphere_name,
                self.sphere_placement,
                self.sphere_color,
                hppfcl.Sphere,
                self.sphere_radius,
            )
            self.geo_ids.append(geo_model.addGeometryObject(self.sphere))

        self.box_x = kwargs.get("box_x", 0.1)
        self.box_y = kwargs.get("box_y", 0.1)
        self.box_z = kwargs.get("box_z", 0)
        self.box = None
        if self.box_x and self.box_y and self.box_z:
            self.box_name = kwargs.get("box_name", f"box_{name}")
            self.box_color = kwargs.get("box_color", WHITE)
            # Placement of the geometry with respect to the joint frame
            self.box_placement = kwargs.get(
                "box_placement", pin.SE3(np.eye(3), np.array([0, 0, self.box_z / 2]))
            )
            self.box = self.__create_geo_obj(
                self.box_name,
                self.box_placement,
                self.box_color,
                hppfcl.Box,
                self.box_x,
                self.box_y,
                self.box_z,
            )
            self.geo_ids.append(geo_model.addGeometryObject(self.box))

    def __create_geo_obj(self, name, placement, color, factory, *nargs, **kwargs):
        return pin.GeometryObject(
            name=name,
            parent_joint=self.id,
            collision_geometry=factory(*nargs, **kwargs),
            placement=placement,
            mesh_path="",
            mesh_scale=np.ones(3),
            override_material=False,
            mesh_color=color,
        )

    def __calculate_placement(self, parent=None):
        if parent is None:
            return pin.SE3.Identity()
        else:
            return pin.SE3(np.eye(3), np.array([0, 0, parent.box_z]))


class Robot(pin.RobotWrapper):
    def __init__(self, root_id=0, show_origin=True):
        model = pin.Model()
        collision_model = pin.GeometryModel()
        visual_model = pin.GeometryModel()
        super().__init__(model, collision_model, visual_model, False)
        self.initViewer()

        self.joints = []
        self.root_id = root_id
        if show_origin:
            self.viewer.gui.addXYZaxis("world/origin", np.zeros(4).tolist(), 0.025, 1.5)

    def add_joint(self, name, geo_model=None, parent=None, **kwargs):
        parent = parent or (self.joints[-1] if self.joints else None)
        geo_model = geo_model or self.visual_model
        self.joints.append(
            Joint(name, self.model, geo_model, parent, self.root_id, **kwargs)
        )
        return self

    def __init_for_demo__(self):
        self.add_joint("joint", box_z=1)

    def rebuildData(self, update_kinematics=True):
        super().rebuildData()
        self.q0 = pin.neutral(self.model)
        self.loadViewerModel()
        if update_kinematics:
            self.forwardKinematics(self.q0)

    def init_for_demo(self, refresh_display=True):
        if not self.nq:
            self.__init_for_demo__()
            self.rebuildData()
            if refresh_display:
                self.display(self.q0)

    @contextmanager
    def capture(self, fname, extension="png"):
        try:
            self.viewer.gui.startCapture(self.viz.windowID, fname, extension)
            yield
        finally:
            # Wait for 0.1 second to make sure that all frames are refreshed and captured.
            time.sleep(0.1)
            self.viewer.gui.stopCapture(self.viz.windowID)


if __name__ == "__main__":
    robot = Robot()
    robot.init_for_demo()
