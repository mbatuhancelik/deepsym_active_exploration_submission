import time
import os
import random

import numpy as np
import pybullet as p

import manipulators
import utils

utils.initialize_env(gui=1)

base = utils.create_object(p.GEOM_BOX, density=0, size=[0.15, 0.15, 0.2], position=[0., 0., 0.2], color=[0.5, 0.5, 0.5, 1.0], with_link=True)
table = utils.create_object(p.GEOM_BOX, density=0, size=[0.5, 1.0, 0.2], position=[0.8, 0, 0.2], color=[1.0, 1.0, 1.0, 1.0])
# walls
utils.create_object(p.GEOM_BOX, density=0, size=[0.5, 0.01, 0.05], position=[0.8, -1, 0.45], color=[1.0, 0.6, 0.6, 1.0])
utils.create_object(p.GEOM_BOX, density=0, size=[0.5, 0.01, 0.05], position=[0.8, 1, 0.45], color=[1.0, 0.6, 0.6, 1.0])
utils.create_object(p.GEOM_BOX, density=0, size=[0.01, 1.0, 0.05], position=[0.3, 0., 0.45], color=[1.0, 0.6, 0.6, 1.0])
utils.create_object(p.GEOM_BOX, density=0, size=[0.01, 1.0, 0.05], position=[1.3, 0., 0.45], color=[1.0, 0.6, 0.6, 1.0])

# agent = manipulators.Manipulator("franka_panda/panda.urdf", position=[0., 0., 0.4], ik_idx=11)
agent = manipulators.Manipulator("ur10e/ur10e.urdf", position=[0., 0., 0.4], ik_idx=10)
constraint_id = p.createConstraint(parentBodyUniqueId=base, parentLinkIndex=0,
                                   childBodyUniqueId=agent.id, childLinkIndex=-1,
                                   jointType=p.JOINT_FIXED, jointAxis=(0, 0, 0),
                                   parentFramePosition=(0, 0, 0),
                                   childFramePosition=(0.0, 0.0, -0.2),
                                   childFrameOrientation=(0, 0, 0, 1))
p.changeConstraint(constraint_id, maxForce=10000)
# robot_start = [0.9, -0.34, 0.35, -2.48, 0.141, 2.16, 1.05, 0, 0]
robot_start = [np.pi/2, -np.pi/3, np.pi/2, -2*np.pi/3, -np.pi/2, 0, 0, 0]
agent.set_joint_position(robot_start, t=2)

# for _ in range(50):
#     obj_type = np.random.choice([p.GEOM_BOX, p.GEOM_SPHERE, p.GEOM_CYLINDER], p=[0.6, 0.1, 0.3])
#     x = np.random.uniform(1.1, 0.4)
#     y = np.random.uniform(-0.9, 0.9)
#     z = np.random.uniform(0.6, 0.7)
#     size = np.random.uniform(0.015, 0.035, (3,)).tolist()
#     if obj_type == p.GEOM_CYLINDER:
#         rotation = [0, 0, 0]
#     else:
#         rotation = np.random.uniform(0, 90, (3,)).tolist()
#     if obj_type == p.GEOM_BOX:
#         if np.random.rand() < 0.5:
#             size = [np.random.uniform(0., 0.2), np.random.uniform(0.01, 0.015),
#                     np.random.uniform(0.015, 0.025)]
#     utils.create_object(obj_type=obj_type, size=size, position=[x, y, z],
#                         rotation=rotation)

p.setJointMotorControlArray(bodyUniqueId=agent.id,
                            jointIndices=agent.joints[-2:],
                            controlMode=p.POSITION_CONTROL,
                            targetPositions=[0.04, 0.04],
                            forces=agent.forces[-2:])
for _ in range(40):
    p.stepSimulation()
    time.sleep(1/240)

utils.create_object(p.GEOM_BOX, [0.025, 0.025, 0.025], [0.6, 0., 0.6])

agent.set_cartesian_position([0.6, 0.0, 0.7], orientation=p.getQuaternionFromEuler([np.pi, 0, 0]), t=2, sleep=True)
agent.set_cartesian_position([0.6, 0.0, 0.41], orientation=p.getQuaternionFromEuler([np.pi, 0, 0]), t=2, sleep=True)

p.setJointMotorControlArray(bodyUniqueId=agent.id,
                            jointIndices=agent.joints[-2:],
                            controlMode=p.POSITION_CONTROL,
                            targetPositions=[0.0, 0.0],
                            forces=agent.forces[-2:])
for _ in range(40):
    p.stepSimulation()
    time.sleep(1/240)

agent.set_cartesian_position([0.6, 0.0, 0.7], orientation=p.getQuaternionFromEuler([np.pi, 0, 0]), t=2, sleep=True)
agent.set_cartesian_position([0.5, 0.0, 0.7], orientation=p.getQuaternionFromEuler([np.pi, 0, 0]), t=2, sleep=True)

p.setJointMotorControlArray(bodyUniqueId=agent.id,
                            jointIndices=agent.joints[-2:],
                            controlMode=p.POSITION_CONTROL,
                            targetPositions=[0.04, 0.04],
                            forces=agent.forces[-2:])
for _ in range(40):
    p.stepSimulation()
    time.sleep(1/240)

while True:
    p.stepSimulation()

# gripper_status = 0.0
# dT = [-0.01, 0.0, 0.01]
# dR = [-np.pi/90, 0, np.pi/90]  # [-2, 0, 2] degrees
# dx = np.random.choice(dT)
# dy = np.random.choice(dT)
# dz = np.random.choice(dT)
# dr = np.random.choice(dR)

# it = 0
# start = time.time()
# while True:
#     if np.random.rand() < 0.1:
#         dx = np.random.choice(dT)
#     if np.random.rand() < 0.1:
#         dy = np.random.choice(dT)
#     if np.random.rand() < 0.1:
#         dz = np.random.choice(dT)
#     if np.random.rand() < 0.1:
#         dr = np.random.choice(dR)

#     position, quaternion = agent.get_tip_pose()
#     position = list(position)
#     rotation = list(p.getEulerFromQuaternion(quaternion))
#     position[0] = np.clip(position[0]+dx, 0.4, 1.1)
#     position[1] = np.clip(position[1]+dy, -0.9, 0.9)
#     position[2] = np.clip(position[2]+dz, 0.42, 0.5)
#     rotation[2] = np.clip(rotation[2]+dr, -np.pi, np.pi)

#     agent.set_cartesian_position(position=position, orientation=p.getQuaternionFromEuler([np.pi, 0, rotation[2]]))

#     if np.random.rand() < 0.01:
#         gripper_status = 0.04 - gripper_status
#         p.setJointMotorControlArray(
#             bodyUniqueId=agent.id,
#             jointIndices=agent.joints[-2:],
#             controlMode=p.POSITION_CONTROL,
#             targetPositions=[gripper_status, gripper_status],
#             forces=agent.forces[-2:])
#         for _ in range(40):
#             p.stepSimulation()
#             # time.sleep(1/240)

#     it += 1
#     p.stepSimulation()
#     # time.sleep(1/240)
#     if it % 1000 == 0:
#         end = time.time()
#         sim_time = it * (1/240)
#         real_time = end - start
#         print(f"RT={real_time:.2f}, ST={sim_time:.2f},  Factor={(sim_time/real_time):.3f}")