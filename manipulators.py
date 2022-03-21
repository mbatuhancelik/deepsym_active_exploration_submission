import time

import pybullet as p


class Manipulator:
    def __init__(self, path, position=(0, 0, 0), orientation=(0, 0, 0, 1), ik_idx=-1):
        self._timestep = p.getPhysicsEngineParameters()["fixedTimeStep"]
        self._freq = int(1. / self._timestep)
        self.id = p.loadURDF(
            fileName=path,
            basePosition=position,
            baseOrientation=orientation,
            flags=p.URDF_USE_SELF_COLLISION)
        self.ik_idx = ik_idx
        self.joints = []
        self.names = []
        self.forces = []

        self.fixed_joints = []
        self.fixed_names = []
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            if info[2] != p.JOINT_FIXED:
                self.joints.append(i)
                self.names.append(info[1])
                self.forces.append(info[10])
            else:
                self.fixed_joints.append(i)
                self.fixed_names.append(info[1])

        self.joints = tuple(self.joints)
        self.names = tuple(self.names)
        self.num_joints = len(self.joints)
        self.debug_params = []
        self.child = None
        self.constraints = []
        for j in self.joints:
            p.enableJointForceTorqueSensor(self.id, j, 1)

    def attach(self, child_body, parent_link, position=(0, 0, 0), orientation=(0, 0, 0, 1), max_force=None):
        self.child = child_body
        constraint_id = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=parent_link,
            childBodyUniqueId=self.child.id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=position,
            childFrameOrientation=orientation)
        if max_force is not None:
            p.changeConstraint(constraint_id, maxForce=max_force)
        self.constraints.append(constraint_id)

    def set_cartesian_position(self, position, orientation=None, t=None, sleep=False):
        target_joints = p.calculateInverseKinematics(
            bodyUniqueId=self.id,
            endEffectorLinkIndex=self.ik_idx,
            targetPosition=position,
            targetOrientation=orientation)
        self.set_joint_position(target_joints[:-2], t=t, sleep=sleep)

    def set_joint_position(self, position, velocity=None, t=None, sleep=False):
        assert len(self.joints) > 0

        if velocity is not None:
            p.setJointMotorControlArray(
                bodyUniqueId=self.id,
                jointIndices=self.joints[:-2],
                controlMode=p.POSITION_CONTROL,
                targetPositions=position,
                targetVelocities=velocity,
                forces=self.forces)
        else:
            p.setJointMotorControlArray(
                bodyUniqueId=self.id,
                jointIndices=self.joints[:-2],
                controlMode=p.POSITION_CONTROL,
                targetPositions=position,
                forces=self.forces[:-2])

        if t is not None:
            iters = int(t*self._freq)
            for _ in range(iters):
                p.stepSimulation()
                if sleep:
                    time.sleep(self._timestep)

    def set_joint_velocity(self, velocity, t=None, sleep=False):
        assert len(self.joints) > 0

        p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=velocity,
            forces=self.forces)

        if t is not None:
            iters = int(t*self._freq)
            for _ in range(iters):
                p.stepSimulation()
                if sleep:
                    time.sleep(self._timestep)

    def set_joint_torque(self, torque, t=None, sleep=False):
        assert len(self.joints) > 0

        p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.joints,
            controlMode=p.TORQUE_CONTROL,
            forces=torque)

        if t is not None:
            iters = int(t*self._freq)
            for _ in range(iters):
                p.stepSimulation()
                if sleep:
                    time.sleep(self._timestep)

    # TODO: make this only joint position, joint velocity etc.
    def get_joint_states(self):
        return p.getJointStates(self.id, self.joints)

    # of IK link
    def get_tip_pose(self):
        result = p.getLinkState(self.id, self.ik_idx)
        return result[0], result[1]

    def add_debug_param(self):
        current_angle = [j[0] for j in self.get_joint_states()]
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.id, self.joints[i])
            low, high = joint_info[8:10]
            self.debug_params.append(p.addUserDebugParameter(self.names[i].decode("utf-8"), low, high, current_angle[i]))

    def update_debug(self):
        target_angles = []
        for param in self.debug_params:
            try:
                angle = p.readUserDebugParameter(param)
                target_angles.append(angle)
            except:
                break
        if len(target_angles) == len(self.joints):
            self.set_joint_position(target_angles)
