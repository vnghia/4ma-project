import time
import warnings
from contextlib import contextmanager

import eigenpy
import hppfcl
import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer

warnings.filterwarnings("ignore")

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

        self.cylinder_radius = kwargs.get("cylinder_radius", self.sphere_radius * 0.75)
        self.cylinder_z = kwargs.get("cylinder_z", 0)
        self.cylinder = None
        if self.cylinder_radius and self.cylinder_z:
            if self.box:
                raise ValueError("Can only add either box, cylinder or capsule !")
            self.cylinder_name = kwargs.get("cylinder_name", f"cylinder_{name}")
            self.cylinder_color = kwargs.get("cylinder_color", WHITE)
            # Same as box
            self.cylinder_placement = kwargs.get(
                "cylinder_placement",
                pin.SE3(np.eye(3), np.array([0, 0, self.cylinder_z / 2])),
            )
            self.cylinder = self.__create_geo_obj(
                self.cylinder_name,
                self.cylinder_placement,
                self.cylinder_color,
                hppfcl.Cylinder,
                self.cylinder_radius,
                self.cylinder_z,
            )
            self.geo_ids.append(geo_model.addGeometryObject(self.cylinder))

        self.capsule_radius = kwargs.get("capsule_radius", self.sphere_radius * 0.75)
        self.capsule_z = kwargs.get("capsule_z", 0)
        self.capsule = None
        if self.capsule_radius and self.capsule_z:
            if self.box or self.cylinder:
                raise ValueError("Can only add either box, cylinder or capsule !")
            self.capsule_name = kwargs.get("capsule_name", f"capsule_{name}")
            self.capsule_color = kwargs.get("capsule_color", WHITE)
            # Same as box
            self.capsule_placement = kwargs.get(
                "capsule_placement",
                pin.SE3(np.eye(3), np.array([0, 0, self.capsule_z / 2])),
            )
            self.capsule = self.__create_geo_obj(
                self.capsule_name,
                self.capsule_placement,
                self.capsule_color,
                hppfcl.Capsule,
                self.capsule_radius,
                self.capsule_z,
            )
            self.geo_ids.append(geo_model.addGeometryObject(self.capsule))

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
            parent_z = 0
            if parent.box_z != 0:
                parent_z = parent.box_z
            elif parent.cylinder_z != 0:
                parent_z = parent.cylinder_z
            else:
                parent_z = parent.capsule_z
            return pin.SE3(np.eye(3), np.array([0, 0, parent_z]))


class Robot(pin.RobotWrapper):
    def __init__(self, root_id=0, show_origin=True, add_plane=True):
        model = pin.Model()
        collision_model = pin.GeometryModel()
        visual_model = pin.GeometryModel()
        super().__init__(model, collision_model, visual_model, False)
        self.__init_viewer__(show_origin)

        self.joints = []
        self.root_id = root_id

        self.plane_name = "plane"
        self.plane = None
        self.plane_geo_id = None
        if add_plane:
            self.plane = pin.GeometryObject(
                name=self.plane_name,
                parent_joint=self.root_id,
                collision_geometry=pin.hppfcl.Plane(0, 0, 1, 0),
                placement=pin.SE3.Identity(),
            )
            self.plane_geo_id = self.collision_model.addGeometryObject(self.plane)

    def __init_viewer__(self, show_origin=True):
        viewer = None
        viz = None
        try:
            import meshcat

            viz = MeshcatVisualizer

            class Visualizer(meshcat.Visualizer):
                Record = {"frame": -1, "animation": None}

                def set_transform(self, matrix=np.eye(4)):
                    animation = self.Record["animation"]
                    if not animation:
                        super().set_transform(matrix)
                    else:
                        frame = self.Record["frame"]
                        with animation.at_frame(self, frame) as f:
                            f.set_transform(matrix)
                        self.Record["frame"] = frame + 1

                def __getitem__(self, path):
                    vis = Visualizer(window=self.window)
                    vis.path = path
                    return vis

                def startCapture(self, rate=30):
                    self.Record["frame"] = 0
                    self.Record["animation"] = meshcat.animation.Animation(
                        default_framerate=rate
                    )

                def stopCapture(self):
                    self.set_animation(self.Record["animation"], play=False)
                    self.Record["frame"] = -1
                    self.Record["animation"] = None

            viewer = Visualizer(zmq_url="tcp://localhost:6000")

        except ImportError:
            viz = GepettoVisualizer

        self.viz = viz(
            self.model,
            self.collision_model,
            self.visual_model,
            False,
            self.data,
            self.collision_data,
            self.visual_data,
        )
        self.viz.initViewer(viewer=viewer)
        self.viz.clean()
        if show_origin and isinstance(self.viz, GepettoVisualizer):
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

    def rebuildData(self):
        self.data, self.collision_data, self.visual_data = pin.createDatas(
            self.model, self.collision_model, self.visual_model
        )

        self.viz.data = self.data
        self.viz.collision_data = self.collision_data
        self.viz.visual_data = self.visual_data

        self.q0 = pin.neutral(self.model)
        self.loadViewerModel()

    def init_for_demo(self, refresh_display=True):
        if not self.nq:
            self.__init_for_demo__()
            self.rebuildData()
            if refresh_display:
                self.display(self.q0)

    @contextmanager
    def capture(self, fname=None, rate=30, extension="png"):
        if isinstance(self.viz, MeshcatVisualizer):
            try:
                self.viewer.startCapture(rate)
                yield
            finally:
                self.viewer.stopCapture()

        elif isinstance(self.viz, GepettoVisualizer):
            fname = fname or self.__class__.__name__
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
