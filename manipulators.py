import time

import numpy as np
import pybullet as p

PI = 3.1415926589793


class Manipulator:
    def __init__(self, path, position=(0, 0, 0), orientation=(0, 0, 0, 1)):
        self.id = p.loadURDF(
            fileName=path,
            basePosition=position,
            baseOrientation=orientation,
            flags=p.URDF_USE_SELF_COLLISION)
        self.joints = []
        self.names = []
        self.forces = []
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            if info[2] == p.JOINT_REVOLUTE:
                self.joints.append(i)
                self.names.append(info[1])
                self.forces.append(info[10])
        self.joints = tuple(self.joints)
        self.names = tuple(self.names)
        self.num_joints = len(self.joints)
        self.debug_params = []

    def move_j(self, position, velocity=None, t=None, sleep=False):
        assert len(self.joints) > 0

        if velocity is not None:
            p.setJointMotorControlArray(
                bodyUniqueId=self.id,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=position,
                targetVelocities=velocity,
                forces=self.forces)
        else:
            p.setJointMotorControlArray(
                bodyUniqueId=self.id,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=position,
                forces=self.forces)

        if t is not None:
            iters = int(t*240)
            for _ in range(iters):
                p.stepSimulation()
                if sleep:
                    time.sleep(1/240)

    def move_v(self, velocity, t=None, sleep=False):
        assert len(self.joints) > 0

        p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=velocity,
            forces=self.forces)

        if t is not None:
            iters = int(t*240)
            for _ in range(iters):
                p.stepSimulation()
                if sleep:
                    time.sleep(1/240)

    def move_t(self, torque, t=None, sleep=False):
        assert len(self.joints) > 0

        p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.joints,
            controlMode=p.TORQUE_CONTROL,
            forces=torque)

        if t is not None:
            iters = int(t*240)
            for _ in range(iters):
                p.stepSimulation()
                if sleep:
                    time.sleep(1/240)

    def get_joint_states(self):
        return p.getJointStates(self.id, self.joints)

    def add_debug_param(self):
        current_angle = [j[0] for j in self.get_joint_states()]
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.id, self.joints[i])
            low, high = joint_info[8:10]
            self.debug_params.append(p.addUserDebugParameter(self.names[i].decode("utf-8"), low, high, current_angle[i]))

    def update_debug(self):
        target_angles = []
        for param in self.debug_params:
            target_angles.append(p.readUserDebugParameter(param))
        self.move_j(target_angles)

    def print_links(self):
        for i in range(6):
            s = p.getDynamicsInfo(self.id, i)
            print("%d - mass: %.3f" % (i, s[0]))
            print("%d - lateral friction: %.3f" % (i, s[1]))
            print("%d - local inertia diag: %.3f %.3f %.3f" % (i, *s[2]))
            print("%d - local inertia pos: %.3f %.3f %.3f" % (i, *s[3]))
            print("%d - local inertia orn: %.3f %.3f %.3f %.3f" % (i, *s[4]))
            print("%d - restitution: %.3f" % (i, s[5]))
            print("%d - rolling friction: %.3f" % (i, s[6]))
            print("%d - spinning friction: %.3f" % (i, s[7]))
            print("%d - contact damping: %.3f" % (i, s[8]))
            print("%d - contact stiffness: %.3f" % (i, s[9]))
            print("%d - body type: %d" % (i, s[10]))
            print("%d - collision margin: %.3f" % (i, s[11]))
            print("-"*20)


class UR10(Manipulator):
    def __init__(self, position=(0, 0, 0), orientation=(0, 0, 0, 1)):
        super(UR10, self).__init__("ur10e/ur10e.urdf", position, orientation)
        self.tool = None
        self.constraints = []
        for j in self.joints:
            p.enableJointForceTorqueSensor(self.id, j, 1)

    def attach_tool(self, tool, position=(0, 0, 0), orientation=(0, 0, 0, 1), max_force=None):
        self.tool = tool
        constraint_id = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=6,
            childBodyUniqueId=self.tool.id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=position,
            childFrameOrientation=orientation)
        if max_force is not None:
            p.changeConstraint(constraint_id, maxForce=max_force)
        self.constraints.append(constraint_id)

    def move_p(self, position, orientation=None, t=None, sleep=False):
        target_joints = p.calculateInverseKinematics(
            bodyUniqueId=self.id,
            endEffectorLinkIndex=6,
            targetPosition=position,
            targetOrientation=orientation)
            # lowerLimits=(-2*PI, -PI, 0, -PI, -PI, -PI),
            # upperLimits=(2*PI, 0, PI, PI, PI, PI),
            # jointRanges=(4*PI, 2*PI, 2*PI, 2*PI, 2*PI, 2*PI),
            # restPoses=(0, -PI/4, PI/2, 0, 0, 0))
        self.move_j(target_joints, t=t, sleep=sleep)


class Robotiq3fGripper(Manipulator):
    def __init__(self, position=(0, 0, 0), orientation=(0, 0, 0, 1)):
        super(Robotiq3fGripper, self).__init__("robotiq-3f-gripper/robotiq-3f-gripper_articulated.urdf", position, orientation)
        for j in self.joints:
            p.enableJointForceTorqueSensor(self.id, j, 1)

    def release(self):
        # p.setJointMotorControlArray(
        #     bodyUniqueId=self.id,
        #     jointIndices=[self.joints[i] for i in [3, 7, 10]],
        #     controlMode=p.POSITION_CONTROL,
        #     targetPositions=[-1.2217, -1.2217, -1.2217],
        #     forces=[100, 100, 100])
        # for _ in range(240):        #     p.stepSimulation()
        #     time.sleep(1/240)
        # p.setJointMotorControlArray(
        #     bodyUniqueId=self.id,
        #     jointIndices=[self.joints[i] for i in [2, 6, 9]],
        #     controlMode=p.POSITION_CONTROL,
        #     targetPositions=[0, 0, 0],
        #     forces=[100, 100, 100])
        # for _ in range(240):
        #     p.stepSimulation()
        #     time.sleep(1/240)
        # p.setJointMotorControlArray(
        #     bodyUniqueId=self.id,
        #     jointIndices=[self.joints[i] for i in [1, 5, 8]],
        #     controlMode=p.POSITION_CONTROL,
        #     targetPositions=[0.0495, 0.0495, 0.0495],
        #     forces=[100, 100, 100])
        # for _ in range(240):
        #     p.stepSimulation()
        #     time.sleep(1/240)
        self.move_j(
          [0.0, 0.0495, 0., -0.255,
           0.0, 0.0495, 0., -0.255,
           0.0495, 0., -0.255], t=1, sleep=True)
        # self.move_v(
        #     [0, -self.vel, 0, 0,
        #      0, -self.vel, 0, 0,
        #         -self.vel, 0, 0], max_force=self.max_force)
        # # self.move_v(
        #     [0, -force, -force, -force,
        #      0, -force, -force, -force,
        #         -force, -force, -force], max_force=self.max_force)

    def grasp(self, threshold):
        state = self.get_joint_states()
        finger_angles = [j[0] for j in state]
        first_set = False
        second_set = False
        third_set = False
        modified = [1, 3, 5, 7, 8, 10]
        while not first_set:
            f1 = np.linalg.norm(state[1][2][:3])
            f2 = np.linalg.norm(state[5][2][:3])
            f3 = np.linalg.norm(state[8][2][:3])
            if (f1 > threshold) and (f2 > threshold) and (f3 > threshold):
                first_set = True
                break

            if f1 < threshold:
                finger_angles[1] += 0.001
                finger_angles[3] -= 0.001
                finger_angles[3] = max(-1.222, min(finger_angles[3], -0.052))
            if f2 < threshold:
                finger_angles[5] += 0.001
                finger_angles[7] -= 0.001
                finger_angles[7] = max(-1.222, min(finger_angles[7], -0.052))
            if f3 < threshold:
                finger_angles[8] += 0.001
                finger_angles[10] -= 0.001
                finger_angles[10] = max(-1.222, min(finger_angles[10], -0.052))

            joints = [self.joints[i] for i in modified]
            angles = [finger_angles[i] for i in modified]
            forces = [100 for i in modified]
            p.setJointMotorControlArray(
                bodyUniqueId=self.id,
                jointIndices=joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=angles,
                forces=forces)

            p.stepSimulation()
            time.sleep(1/240)
            state = self.get_joint_states()
            print(f"{np.linalg.norm(state[1][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[5][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[8][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[2][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[6][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[9][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[3][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[7][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[10][2][:3]):.1f}")

        modified = [2, 3, 6, 7, 9, 10]
        while not second_set:
            f1 = np.linalg.norm(state[2][2][:3])
            f2 = np.linalg.norm(state[6][2][:3])
            f3 = np.linalg.norm(state[9][2][:3])
            if (f1 > threshold) and (f2 > threshold) and (f3 > threshold):
                second_set = True
                break

            if f1 < threshold:
                finger_angles[2] += 0.001
                finger_angles[3] -= 0.001
                finger_angles[3] = max(-1.222, min(finger_angles[3], -0.052))
            if f2 < threshold:
                finger_angles[6] += 0.001
                finger_angles[7] -= 0.001
                finger_angles[7] = max(-1.222, min(finger_angles[7], -0.052))
            if f3 < threshold:
                finger_angles[9] += 0.001
                finger_angles[10] -= 0.001
                finger_angles[10] = max(-1.222, min(finger_angles[10], -0.052))

            joints = [self.joints[i] for i in modified]
            angles = [finger_angles[i] for i in modified]
            forces = [100 for i in modified]
            p.setJointMotorControlArray(
                bodyUniqueId=self.id,
                jointIndices=joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=angles,
                forces=forces)

            p.stepSimulation()
            time.sleep(1/240)
            state = self.get_joint_states()
            print(f"{np.linalg.norm(state[1][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[5][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[8][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[2][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[6][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[9][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[3][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[7][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[10][2][:3]):.1f}")

        modified = [3, 7, 10]
        while not third_set:
            f1 = np.linalg.norm(state[3][2][:3])
            f2 = np.linalg.norm(state[7][2][:3])
            f3 = np.linalg.norm(state[10][2][:3])
            if (f1 > threshold) and (f2 > threshold) and (f3 > threshold):
                third_set = True
                break

            if f1 < threshold:
                finger_angles[3] += 0.001
            if f2 < threshold:
                finger_angles[7] += 0.001
            if f3 < threshold:
                finger_angles[10] += 0.001

            joints = [self.joints[i] for i in modified]
            angles = [finger_angles[i] for i in modified]
            forces = [100 for i in modified]
            p.setJointMotorControlArray(
                bodyUniqueId=self.id,
                jointIndices=joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=angles,
                forces=forces)

            p.stepSimulation()
            time.sleep(1/240)
            state = self.get_joint_states()
            print(f"{np.linalg.norm(state[1][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[5][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[8][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[2][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[6][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[9][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[3][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[7][2][:3]):.1f}, "
                  f"{np.linalg.norm(state[10][2][:3]):.1f}")
