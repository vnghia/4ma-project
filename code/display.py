import os
import time
from contextlib import contextmanager

import pinocchio as pin
from gepetto import corbaserver


class Display:
    def __init__(self, window_name="pinocchio") -> None:
        # Start server if needed
        corbaserver.start_server()

        # Create gui client for interact with the server
        self.gui = corbaserver.gui_client(window_name)

        self.window_id = self.gui.getWindowID(window_name)
        self.root_scene = "world"

        # If "world" is already exists, delete it
        if self.gui.nodeExists(self.root_scene):
            self.gui.deleteNode(self.root_scene, True)

        # (Re)create our root scene
        self.gui.createSceneWithFloor(self.root_scene)
        self.gui.addSceneToWindow(self.root_scene, self.window_id)

    @contextmanager
    def capture(self, fname, extension="png"):
        try:
            self.gui.startCapture(self.window_id, fname, extension)
            yield
        finally:
            # Wait for 0.1 second to make sure that all frames are refreshed and captured.
            time.sleep(0.1)
            self.gui.stopCapture(self.window_id)

    def no_floor(self):
        # Turn off the rendering of the floor
        self.gui.setVisibility(os.path.join(self.root_scene, "floor"), "OFF")

    def place(self, name, pos):
        # Place an object whose name is "name" in the position "pos"
        self.gui.applyConfiguration(name, pin.SE3ToXYZQUATtuple(pos))

    def places(self, names, poss):
        # Place a list of objects whose name are "names" in the position "poss".
        # This function should be a little bit faster than calling `place` for one element.
        # This function does not convert `SE3ToXYZQUATtuple` !
        self.gui.applyConfigurations(names, poss)

    def refresh(self):
        # Refresh the gui to show new objects
        self.gui.refresh()


if __name__ == "__main__":
    Display()
