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
        print("You should provide at least one target.")

GUI = 1
if GUI:
    physicsClient = p.connect(p.GUI)
else:
    physicsClient = p.connect(p.DIRECT)
    egl = pkgutil.get_loader("eglRenderer")
    p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)

planeId = p.loadURDF("plane.urdf")
robot = p.loadURDF("ur10e/ur10e.urdf", [0, 0, 0.6])

numJoints = p.getNumJoints(robot)
robot_joints = []
for i in range(numJoints):
    robot_joints.append(p.addUserDebugParameter("j"+str(i), -np.pi, np.pi, 0))

base = p.loadURDF("robotiq-3f-gripper/robotiq-3f-gripper_articulated.urdf")
constraint_id = p.createConstraint(
    parentBodyUniqueId=robot,
    parentLinkIndex=6,
    childBodyUniqueId=base,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=(0, 0, 0),
    parentFramePosition=(0, 0, 0),
    childFramePosition=(0, -0.04, 0),
    childFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]))

numJoints2 = p.getNumJoints(base)
gripper_joints = []
for i in range(numJoints2):
    gripper_joints.append(p.addUserDebugParameter("g"+str(i), -np.pi, np.pi, 0))

it = 0
quat = Rotation.from_euler("zyx", [0, 0, -90], degrees=True).as_quat()
while True:
    # joints = np.zeros(numJoints)
    # for i in range(numJoints):
    #     joints[i] = p.readUserDebugParameter(robot_joints[i])
    # move(robot, targetPositions=joints)

    # joints = np.zeros(numJoints2)
    # for i in range(numJoints2):
    #     joints[i] = p.readUserDebugParameter(gripper_joints[i])
    # move(base, targetPositions=joints)
    # p.stepSimulation()

    p.stepSimulation()
    position, orientation = p.getBasePositionAndOrientation(robot)
    jointInfo = p.getJointStates(robot, range(numJoints))
    jointPosition = [jointInfo[i][0] for i in range(numJoints)]
    jointVelocity = [jointInfo[i][1] for i in range(numJoints)]

    angle = (it / 240) * np.pi * 2
    target_joints = p.calculateInverseKinematics(
        bodyUniqueId=robot,
        endEffectorLinkIndex=6,
        targetPosition=[0.6 + np.cos(angle)*0.1, np.sin(angle)*0.1, 1],
        targetOrientation=quat,
        lowerLimits=[-2*np.pi, -np.pi, 0, -np.pi, -np.pi, -np.pi],
        upperLimits=[2*np.pi, 0, np.pi, np.pi, np.pi, np.pi],
        jointRanges=[4*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi],
        restPoses=[0, -np.pi/4, np.pi/2, 0, 0, 0]
    )
    move(robot, targetPositions=target_joints)
    it += 1
