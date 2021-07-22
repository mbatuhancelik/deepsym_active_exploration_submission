import time
import os
import random
import pkgutil

import numpy as np
import pybullet as p
import pybullet_data

from scipy.spatial.transform import Rotation

def move(robot, targetPositions=None, targetVelocities=None):
    if targetPositions is None:
        p.setJointMotorControlArray(
            bodyUniqueId=robot,
            jointIndices=range(len(targetVelocities)),
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=targetVelocities,
        )
    elif targetVelocities is None:
        p.setJointMotorControlArray(
            bodyUniqueId=robot,
            jointIndices=range(len(targetPositions)),
            controlMode=p.POSITION_CONTROL,
            targetPositions=targetPositions,
        )
    else:
        print("Birini vermek zorundasin!")

GUI = 1

if GUI:
    physicsClient = p.connect(p.GUI)
else:
    physicsClient = p.connect(p.DIRECT)
    egl = pkgutil.get_loader("eglRenderer")
    p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
# p.setRealTimeSimulation(1)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")
startOrientation = p.getQuaternionFromEuler([0,0,0])
objId = p.loadSDF("kuka_iiwa/kuka_with_gripper2.sdf")[0]
p.resetBasePositionAndOrientation(objId, [0, 0, 0.6], [0, 0, 0, 1])
# objId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0.6])
# objId = p.loadURDF("ur5.urdf", [0, 0, 0.6])
p.loadURDF("table/table.urdf", [0, -0.7, 0.])

# p.loadURDF("cube_small.urdf", [0, -0.7, 1.])
viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[0, -0.7, 1.7],
    cameraTargetPosition=[0, -0.7, 0],
    cameraUpVector=[0, 1, 0]
    )
projectionMatrix = p.computeProjectionMatrixFOV(
    fov=45,
    aspect=1.0,
    nearVal=0.01,
    farVal=5
    )

collisionShapeId = p.createCollisionShape(
    shapeType=p.GEOM_BOX,
    # radius=0.05,
    halfExtents=[0.05, 0.05, 0.05]
)
visualShapeId = p.createVisualShape(
    shapeType=p.GEOM_BOX,
    # radius=0.05,
    halfExtents=[0.05, 0.05, 0.05],
    rgbaColor=[1, 0, 0, 1],
    specularColor=[0.4, 0.4, 0.0]
)

p.createMultiBody(
    baseMass=0.05,
    baseCollisionShapeIndex=collisionShapeId,
    baseVisualShapeIndex=visualShapeId,
    basePosition=[0, -0.7, 0.65])
p.createMultiBody(
    baseMass=0.05,
    baseCollisionShapeIndex=collisionShapeId,
    baseVisualShapeIndex=visualShapeId,
    basePosition=[0.3, -0.7, 0.65])
p.createMultiBody(
    baseMass=0.05,
    baseCollisionShapeIndex=collisionShapeId,
    baseVisualShapeIndex=visualShapeId,
    basePosition=[-0.3, -0.7, 0.65])

# for i in range(20):
#     p.createMultiBody(
#         baseMass=0.01,
#         baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.01),
#         basePosition=[0.5, -0.7, 0.66+i*0.02]
#     )

numJoints = p.getNumJoints(objId)
print(f"This have {numJoints} joints.")
for i in range(numJoints):
    print(p.getJointInfo(objId, i))
# exit()

rnd = np.random.randn(numJoints)*0.1
quat = Rotation.from_euler("zyx", [0, 180, 0], degrees=True).as_quat()
target_joints = p.calculateInverseKinematics(
    bodyUniqueId=objId,
    endEffectorLinkIndex=7,
    targetPosition=[0.4, -0.7, 0.9],
    targetOrientation=quat
)
move(objId, targetPositions=target_joints)

it = 0
while True:
    p.stepSimulation()
    position, orientation = p.getBasePositionAndOrientation(objId)
    jointInfo = p.getJointStates(objId, range(numJoints))
    jointPosition = [jointInfo[i][0] for i in range(numJoints)]
    jointVelocity = [jointInfo[i][1] for i in range(numJoints)]

    angle = (it / 240) * np.pi * 2
    target_joints = p.calculateInverseKinematics(
        bodyUniqueId=objId,
        endEffectorLinkIndex=7,
        targetPosition=[np.cos(angle)*0.1, -0.6+np.sin(angle)*0.1, 0.9],
        targetOrientation=quat
    )
    move(objId, targetPositions=target_joints)
    it += 1

    # noise = np.random.randn(numJoints)*0.01
    # rnd = rnd + noise
    # move(objId, targetPositions=rnd)

    p.getCameraImage(width=128, height=128, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix)
    # print(", ".join(["%.3f" % jointPosition[i] for i in range(numJoints)]))
p.disconnect()
