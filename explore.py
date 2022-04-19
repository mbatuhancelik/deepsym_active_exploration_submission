import time
import os
import argparse

import numpy as np
import pybullet as p
import torch

import manipulators
import utils

parser = argparse.ArgumentParser("Explore environment.")
parser.add_argument("-N", help="number of interactions", type=int, required=True)
args = parser.parse_args()

utils.initialize_env(gui=0)
env_objects = utils.create_tabletop()

agent = manipulators.Manipulator("ur10e/ur10e.urdf", position=[0., 0., 0.4], ik_idx=10)
# attach UR10 to a base
constraint_id = p.createConstraint(parentBodyUniqueId=env_objects["base"], parentLinkIndex=0,
                                   childBodyUniqueId=agent.id, childLinkIndex=-1,
                                   jointType=p.JOINT_FIXED, jointAxis=(0, 0, 0),
                                   parentFramePosition=(0, 0, 0),
                                   childFramePosition=(0.0, 0.0, -0.2),
                                   childFrameOrientation=(0, 0, 0, 1))
p.changeConstraint(constraint_id, maxForce=10000)
# force grippers to act in sync.
c = p.createConstraint(agent.id, 8, agent.id, 9,
                       jointType=p.JOINT_GEAR,
                       jointAxis=[1, 0, 0],
                       parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

robot_start = [np.pi/2, -np.pi/3, np.pi/2, -2*np.pi/3, -np.pi/2, 0]
agent.set_joint_position(robot_start, t=2)
agent.set_cartesian_position([0.6, 0.0, 0.7], orientation=p.getQuaternionFromEuler([np.pi, 0, 0]), t=2, sleep=True)

# generate random objects
for _ in range(10):
    # obj_type = np.random.choice([p.GEOM_BOX, p.GEOM_SPHERE, p.GEOM_CYLINDER], p=[0.6, 0.1, 0.3])
    obj_type = p.GEOM_BOX
    x = np.random.uniform(1.1, 0.4)
    y = np.random.uniform(-0.9, 0.9)
    z = np.random.uniform(0.6, 0.7)
    size = np.random.uniform(0.015, 0.035, (3,)).tolist()
    if obj_type == p.GEOM_CYLINDER:
        rotation = [0, 0, 0]
        color = [1.0, 1.0, 0.0, 1.0]
    else:
        rotation = np.random.uniform(0, 90, (3,)).tolist()
        color = [0.0, 0.0, 1.0, 1.0]
    if obj_type == p.GEOM_BOX:
        color = [1.0, 0.0, 0.0, 1.0]
        # if np.random.rand() < 0.5:
        #     size = [np.random.uniform(0., 0.2), np.random.uniform(0.01, 0.015),
        #             np.random.uniform(0.015, 0.025)]
        #     color = [0.0, 1.0, 1.0, 1.0]
    utils.create_object(obj_type=obj_type, size=size, position=[x, y, z],
                        rotation=rotation, color=color)

for _ in range(40):
    p.stepSimulation()
    time.sleep(1/240)

agent.open_gripper()
for _ in range(40):
    p.stepSimulation()
    time.sleep(1/240)
agent.set_cartesian_position([0.6, 0.0, 0.41], orientation=p.getQuaternionFromEuler([np.pi, 0, 0]), t=2, sleep=True)
agent.close_gripper()

for _ in range(40):
    p.stepSimulation()
    time.sleep(1/240)

DATA_SIZE = args.N
state_vector = torch.zeros(DATA_SIZE, 3, 128, 128, dtype=torch.uint8)
action_vector = torch.zeros(DATA_SIZE, 13, dtype=torch.uint8)

gripper_open = True
dT = [-0.01, 0.0, 0.01]
dR = [-np.pi/90, 0, np.pi/90]  # [-2, 0, 2] degrees
d_action = np.zeros(5, dtype=np.int32)
d_action[:4] = np.random.randint(0, 3, (4,))
dx = dT[d_action[0]]
dy = dT[d_action[1]]
dz = dT[d_action[2]]
dr = dR[d_action[3]]
eye3 = np.eye(3)

frame = 0
it = 0
start = time.time()
while it < DATA_SIZE:
    if frame % 24 == 0:
        rgb = utils.get_image([2.0, 0.0, 2.5], [0.8, 0., 0.4], [0, 0, 1], 128, 128)[:, :, :3]
        state_vector[it] = torch.tensor(np.transpose(rgb, (2, 0, 1)))

        if np.random.rand() < 0.5:
            d_action[0] = np.random.randint(0, 3)
        if np.random.rand() < 0.5:
            d_action[1] = np.random.randint(0, 3)
        if np.random.rand() < 0.5:
            d_action[2] = np.random.randint(0, 3)
        if np.random.rand() < 0.5:
            d_action[3] = np.random.randint(0, 3)

        if np.random.rand() < 0.50:
            d_action[4] = 1
            if gripper_open:
                agent.close_gripper(t=40)
                gripper_open = False
            else:
                agent.open_gripper(t=40)
                gripper_open = True
        else:
            d_action[4] = 0

        action_vector[it] = torch.tensor(np.concatenate([eye3[d_action[0]], eye3[d_action[1]], eye3[d_action[2]], eye3[d_action[3]], [d_action[4]]], axis=0))

        dx = dT[d_action[0]]
        dy = dT[d_action[1]]
        dz = dT[d_action[2]]
        dr = dR[d_action[3]]

        it += 1
        if it % 1000 == 0:
            print(f"{100*it/DATA_SIZE:.1f}% completed.")

    position, quaternion = agent.get_tip_pose()
    position = list(position)
    rotation = list(p.getEulerFromQuaternion(quaternion))
    position[0] = np.clip(position[0]+dx, 0.4, 1.1)
    position[1] = np.clip(position[1]+dy, -0.9, 0.9)
    position[2] = np.clip(position[2]+dz, 0.42, 0.5)
    rotation[2] = np.clip(rotation[2]+dr, -np.pi, np.pi)
    agent.set_cartesian_position(position=position, orientation=p.getQuaternionFromEuler([np.pi, 0, rotation[2]]))
    p.stepSimulation()
    frame += 1

    if frame % 1000 == 0:
        end = time.time()
        sim_time = frame * (1/240)
        real_time = end - start
        start = end
        frame = 0
        print(f"RT={real_time:.2f}, ST={sim_time:.2f},  Factor={(sim_time/real_time):.3f}")

if not os.path.exists("data"):
    os.makedirs("data")

torch.save(state_vector, "data/state.pt")
torch.save(action_vector, "data/action.pt")
