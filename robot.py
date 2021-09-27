import time
import os
import random

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

import manipulators
import utils

utils.initialize_env(gui=1)

ur10 = manipulators.UR10([0, 0, 0.65])
gripper = manipulators.Robotiq3fGripper()
ur10.print_links()
gripper.print_links()

ur10.attach_tool(tool=gripper, position=(0, -0.04, 0))
robot_start = [np.pi/2, -np.pi/3, np.pi/2, -2*np.pi/3, -np.pi/2, 0]
ur10.move_j(robot_start, t=2, sleep=True)

p.loadURDF("table/table.urdf", [0, 0.7, 0])

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
    baseMass=1,
    baseCollisionShapeIndex=collisionShapeId,
    baseVisualShapeIndex=visualShapeId,
    basePosition=[0, 0.7, 0.65])
p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=collisionShapeId,
    baseVisualShapeIndex=visualShapeId,
    basePosition=[0.3, 0.7, 0.65])
p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=collisionShapeId,
    baseVisualShapeIndex=visualShapeId,
    basePosition=[-0.3, 0.7, 0.65])

ur10.tool.release()
ur10.move_p([-0.3, 0.7, 1.2], p.getQuaternionFromEuler([-np.pi/2, 0, np.pi]), t=2, sleep=True)
ur10_states = ur10.get_joint_states()
print(" ".join(["%.2f" % j[3] for j in ur10_states]))
gripper_states = ur10.tool.get_joint_states()
print(" ".join(["%.2f" % j[3] for j in gripper_states]))
height = 1.2
force = np.linalg.norm(ur10.get_joint_states()[5][2][:3])
while force < 30:
    ur10.move_p([-0.3, 0.7, height], p.getQuaternionFromEuler([-np.pi/2, 0, np.pi]), t=1/240, sleep=True)
    print("h: %.3f, F=%.3f" % (height, force))
    height -= 0.001
    force = np.linalg.norm(ur10.get_joint_states()[5][2][:3])
ur10.move_p([-0.3, 0.7, height+0.02], p.getQuaternionFromEuler([-np.pi/2, 0, np.pi]), t=1, sleep=True)

threshold = 60
ur10.tool.grasp(threshold)

ur10.move_p([-0.3, 0.7, 1.0], p.getQuaternionFromEuler([-np.pi/2, 0, np.pi]), t=3, sleep=True)
ur10.move_p([0.3, 0.7, 1.0], p.getQuaternionFromEuler([-np.pi/2, 0, np.pi]), t=3, sleep=True)
ur10.tool.release()
# ur10.move_p([-0.3, 0.7, 0.83], p.getQuaternionFromEuler([-np.pi/2, 0, 0]), t=2, sleep=True)
# states = ur10.get_joint_states()
# print(" ".join(["%.2f" % j[3] for j in states]))
# states = ur10.tool.get_joint_states()
# print(" ".join(["%.2f" % j[3] for j in states]))

ur10.add_debug_param()
ur10.tool.add_debug_param()
# p.setRealTimeSimulation(1)
while True:
    # states = ur10.get_joint_states()
    # print(" ".join(["%.2f" % j[3] for j in states]))
    print("="*24)
    states = ur10.tool.get_joint_states()
    for i, j in enumerate(states):
        print(f"{ur10.tool.names[i].decode('utf-8')}: F={np.linalg.norm(j[2][:3]):.2f}, M={np.linalg.norm(j[2][3:]):.2f}")
    # print(" ".join(["%.2f" % j[3] for j in states]))
    ur10.update_debug()
    ur10.tool.update_debug()
    time.sleep(1/240)
    p.stepSimulation()
