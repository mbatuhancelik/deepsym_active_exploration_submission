import pkgutil

import pybullet as p
import pybullet_data
import numpy as np

def initialize_env(gui=1, timestep=1/240):
    if gui:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
        egl = pkgutil.get_loader("eglRenderer")
        if egl is not None:
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
    p.setTimeStep(timestep)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.807)
    p.loadURDF("plane.urdf")
    return physicsClient


def create_object(obj_type, size, position, rotation=[0, 0, 0], density=0.1, color=None, with_link=False):
    collisionId = -1
    visualId = -1
    volume = None

    if obj_type == p.GEOM_SPHERE:
        volume = 4/3 * np.pi * (size[0]**3)
        collisionId = p.createCollisionShape(shapeType=obj_type, radius=size[0])

        if color == "random":
            color = np.random.rand(3).tolist() + [1]
            visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], rgbaColor=color)
        elif color is not None:
            visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], rgbaColor=color)

    elif obj_type in [p.GEOM_CAPSULE, p.GEOM_CYLINDER]:
        volume = np.pi * (size[0]**2) * size[1]  # this is approx. for capsule
        collisionId = p.createCollisionShape(shapeType=obj_type, radius=size[0], height=size[1])

        if color == "random":
            color = np.random.rand(3).tolist() + [1]
            visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], length=size[1], rgbaColor=color)
        elif color is not None:
            visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], length=size[1], rgbaColor=color)

    elif obj_type == p.GEOM_BOX:
        volume = size[0] * size[1] * size[2]
        collisionId = p.createCollisionShape(shapeType=obj_type, halfExtents=size)

        if color == "random":
            color = np.random.rand(3).tolist() + [1]
            visualId = p.createVisualShape(shapeType=obj_type, halfExtents=size, rgbaColor=color)
        elif color is not None:
            visualId = p.createVisualShape(shapeType=obj_type, halfExtents=size, rgbaColor=color)

    mass = volume * density
    if with_link:
        obj_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1,
                                   basePosition=position, baseOrientation=p.getQuaternionFromEuler(rotation),
                                   linkMasses=[mass], linkCollisionShapeIndices=[collisionId], linkVisualShapeIndices=[visualId],
                                   linkPositions=[[0, 0, 0]], linkOrientations=[[0, 0, 0, 1]],
                                   linkInertialFramePositions=[[0, 0, 0]], linkInertialFrameOrientations=[[0, 0, 0, 1]],
                                   linkParentIndices=[0], linkJointTypes=[p.JOINT_FIXED], linkJointAxis=[[0, 0, 0]])
    else:
        obj_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collisionId, baseVisualShapeIndex=visualId,
                                   basePosition=position, baseOrientation=p.getQuaternionFromEuler(rotation))

    return obj_id