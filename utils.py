import pkgutil

import pybullet as p
import pybullet_data

def initialize_env(gui=1):
    if gui:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
        egl = pkgutil.get_loader("eglRenderer")
        p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.807)
    p.loadURDF("plane.urdf")
    return physicsClient
