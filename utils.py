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


def create_object(obj_type, size, position, rotation=[0, 0, 0], mass=1, color=None, with_link=False):
    collisionId = -1
    visualId = -1

    if obj_type == p.GEOM_SPHERE:
        collisionId = p.createCollisionShape(shapeType=obj_type, radius=size[0])

        if color == "random":
            color = np.random.rand(3).tolist() + [1]
            visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], rgbaColor=color)
        elif color is not None:
            visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], rgbaColor=color)

    elif obj_type in [p.GEOM_CAPSULE, p.GEOM_CYLINDER]:
        collisionId = p.createCollisionShape(shapeType=obj_type, radius=size[0], height=size[1])

        if color == "random":
            color = np.random.rand(3).tolist() + [1]
            visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], length=size[1], rgbaColor=color)
        elif color is not None:
            visualId = p.createVisualShape(shapeType=obj_type, radius=size[0], length=size[1], rgbaColor=color)

    elif obj_type == p.GEOM_BOX:
        collisionId = p.createCollisionShape(shapeType=obj_type, halfExtents=size)

        if color == "random":
            color = np.random.rand(3).tolist() + [1]
            visualId = p.createVisualShape(shapeType=obj_type, halfExtents=size, rgbaColor=color)
        elif color is not None:
            visualId = p.createVisualShape(shapeType=obj_type, halfExtents=size, rgbaColor=color)

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


def create_tabletop():
    objects = {}
    objects["base"] = create_object(p.GEOM_BOX, mass=0, size=[0.15, 0.15, 0.2],
                                    position=[0., 0., 0.2], color=[0.5, 0.5, 0.5, 1.0], with_link=True)
    objects["table"] = create_object(p.GEOM_BOX, mass=0, size=[0.5, 1.0, 0.2],
                                     position=[0.8, 0, 0.2], color=[1.0, 1.0, 1.0, 1.0])
    # walls
    objects["wall1"] = create_object(p.GEOM_BOX, mass=0, size=[0.5, 0.01, 0.05],
                                     position=[0.8, -1, 0.45], color=[1.0, 0.6, 0.6, 1.0])
    objects["wall2"] = create_object(p.GEOM_BOX, mass=0, size=[0.5, 0.01, 0.05],
                                     position=[0.8, 1, 0.45], color=[1.0, 0.6, 0.6, 1.0])
    objects["wall3"] = create_object(p.GEOM_BOX, mass=0, size=[0.01, 1.0, 0.05],
                                     position=[0.3, 0., 0.45], color=[1.0, 0.6, 0.6, 1.0])
    objects["wall4"] = create_object(p.GEOM_BOX, mass=0, size=[0.01, 1.0, 0.05],
                                     position=[1.3, 0., 0.45], color=[1.0, 0.6, 0.6, 1.0])
    return objects
