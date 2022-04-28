import pybullet_data
import numpy as np

import utils
import manipulators


class BlocksWorld:
    def __init__(self, gui=0):
        self._p = utils.connect(gui)
        self.reset()

    def reset(self):
        self._p.resetSimulation()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.807)
        self._p.loadURDF("plane.urdf")

        self.env_dict = utils.create_tabletop(self._p)
        self.obj_dict = {}
        self.agent = manipulators.Manipulator(p=self._p, path="ur10e/ur10e.urdf", position=[0., 0., 0.4], ik_idx=10)
        tool_constraint = self._p.createConstraint(parentBodyUniqueId=self.env_dict["base"], parentLinkIndex=0,
                                                   childBodyUniqueId=self.agent.id, childLinkIndex=-1,
                                                   jointType=self._p.JOINT_FIXED, jointAxis=(0, 0, 0),
                                                   parentFramePosition=(0, 0, 0),
                                                   childFramePosition=(0.0, 0.0, -0.2),
                                                   childFrameOrientation=(0, 0, 0, 1))

        self._p.changeConstraint(tool_constraint, maxForce=10000)
        # force grippers to act in sync
        mimic_constraint = self._p.createConstraint(self.agent.id, 8, self.agent.id, 9,
                                                    jointType=self._p.JOINT_GEAR,
                                                    jointAxis=[1, 0, 0],
                                                    parentFramePosition=[0, 0, 0],
                                                    childFramePosition=[0, 0, 0])
        self._p.changeConstraint(mimic_constraint, gearRatio=-1, erp=0.1, maxForce=50)

        self.init_agent_pose(t=1)
        self.init_objects()
        self._step(40)
        self.agent.open_gripper(1, sleep=True)

    def init_agent_pose(self, t=None, sleep=False, traj=False):
        angles = [-0.294, -1.650, 2.141, -2.062, -1.572, 1.277]
        self.agent.set_joint_position(angles, t=t, sleep=sleep, traj=traj)

    def init_objects(self):
        for i in range(10):
            obj_type = np.random.choice([self._p.GEOM_BOX, self._p.GEOM_SPHERE, self._p.GEOM_CYLINDER], p=[0.6, 0.1, 0.3])
            # obj_type = self._p.GEOM_BOX
            x = np.random.uniform(0.5, 1.0)
            y = np.random.uniform(-0.4, 0.4)
            z = np.random.uniform(0.6, 0.65)
            size = np.random.uniform(0.015, 0.035, (3,)).tolist()
            # size = [0.025, 0.025, 0.025]
            if obj_type == self._p.GEOM_CYLINDER:
                rotation = [0, 0, 0]
                color = [1.0, 1.0, 0.0, 1.0]
            else:
                rotation = np.random.uniform(0, 90, (3,)).tolist()
                # rotation = [0, 0, 0]
                color = [0.0, 0.0, 1.0, 1.0]
            if obj_type == self._p.GEOM_BOX:
                color = [1.0, 0.0, 0.0, 1.0]
                if np.random.rand() < 0.5:
                    size = [np.random.uniform(0., 0.2), np.random.uniform(0.01, 0.015),
                            np.random.uniform(0.015, 0.025)]
                    color = [0.0, 1.0, 1.0, 1.0]
            self.obj_dict[i] = utils.create_object(p=self._p, obj_type=obj_type, size=size, position=[x, y, z],
                                                   rotation=rotation, color="random", mass=0.1)

    def state(self):
        rgb, depth, seg = utils.get_image(p=self._p, eye_position=[1.5, 0.0, 1.5], target_position=[0.9, 0., 0.4],
                                          up_vector=[0, 0, 1], height=256, width=256)
        return rgb[:, :, :3], depth, seg

    def step(self, from_loc, to_loc, sleep=False):
        traj_time = 0.5
        self.agent.set_cartesian_position(from_loc[:2]+[0.75], orientation=self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=traj_time, traj=True, sleep=sleep)
        self.agent.set_cartesian_position(from_loc, orientation=self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=traj_time, traj=True, sleep=sleep)
        self.agent.close_gripper(traj_time, sleep=sleep)
        self.agent.set_cartesian_position(from_loc[:2]+[0.75], orientation=self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=traj_time, traj=True, sleep=sleep)
        self.agent.set_cartesian_position(to_loc, orientation=self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=traj_time, traj=True, sleep=sleep)
        self.agent._waitsleep(0.5, sleep=sleep)
        self.agent.open_gripper(traj_time, sleep=sleep)
        self.init_agent_pose(t=1.0, sleep=sleep)

    def _step(self, count=1):
        for _ in range(count):
            self._p.stepSimulation()
