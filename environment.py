import torch
import copy
import pybullet
import pybullet_data
import numpy as np

import utils
import manipulators


class GenericEnv:
    def __init__(self, gui=0, seed=None):
        self._p = utils.connect(gui)
        self.gui = gui
        self.reset(seed=seed)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self._p.resetSimulation()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.807)
        self._p.loadURDF("plane.urdf")

        self.env_dict = utils.create_tabletop(self._p)
        self.agent = manipulators.Manipulator(p=self._p, path="ur10e/ur10e.urdf", position=[0., 0., 0.4], ik_idx=10)
        base_constraint = self._p.createConstraint(parentBodyUniqueId=self.env_dict["base"], parentLinkIndex=0,
                                                   childBodyUniqueId=self.agent.id, childLinkIndex=-1,
                                                   jointType=self._p.JOINT_FIXED, jointAxis=(0, 0, 0),
                                                   parentFramePosition=(0, 0, 0),
                                                   childFramePosition=(0.0, 0.0, -0.2),
                                                   childFrameOrientation=(0, 0, 0, 1))
        self._p.changeConstraint(base_constraint, maxForce=10000)
        # force grippers to act in sync
        mimic_constraint = self._p.createConstraint(self.agent.id, 8, self.agent.id, 9,
                                                    jointType=self._p.JOINT_GEAR,
                                                    jointAxis=[1, 0, 0],
                                                    parentFramePosition=[0, 0, 0],
                                                    childFramePosition=[0, 0, 0])
        self._p.changeConstraint(mimic_constraint, gearRatio=-1, erp=0.1, maxForce=50)

    def init_agent_pose(self, t=None, sleep=False, traj=False):
        angles = [-0.294, -1.950, 2.141, -2.062, -1.572, 1.277]
        self.agent.set_joint_position(angles, t=t, sleep=sleep, traj=traj)

    def state_obj_poses(self):
        N_obj = len(self.obj_dict)
        pose = np.zeros((N_obj, 7), dtype=np.float32)
        for i in range(N_obj):
            position, quaternion = self._p.getBasePositionAndOrientation(self.obj_dict[i])
            pose[i][:3] = position
            pose[i][3:] = quaternion
        return pose

    def _step(self, count=1):
        for _ in range(count):
            self._p.stepSimulation()

    def __del__(self):
        self._p.disconnect()


class BlocksWorld(GenericEnv):
    def __init__(self, gui=0, seed=None, min_objects=2, max_objects=5):
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.num_objects = None
        super(BlocksWorld, self).__init__(gui=gui, seed=seed)

    def reset(self, seed=None):
        super(BlocksWorld, self).reset(seed=seed)

        self.obj_dict = {}
        self.init_agent_pose(t=1)
        self.init_objects()
        self._step(40)
        self.agent.open_gripper(1, sleep=True)

    def delete_objects(self):
        self.current_obj_locs = [[[] for _ in self.y_locs] for _ in self.x_locs]
        for key in self.obj_dict:
            obj_id = self.obj_dict[key]
            self._p.removeBody(obj_id)
        self.obj_dict = {}
    def reset_objects(self):
        self.delete_objects()
        self.init_objects()
        self._step(240)

    def reset_object_poses(self):
        for key in self.obj_dict:
            x = np.random.uniform(0.5, 1.0)
            y = np.random.uniform(-0.4, 0.4)
            z = np.random.uniform(0.6, 0.65)
            quat = pybullet.getQuaternionFromEuler(np.random.uniform(0, 90, (3,)).tolist())

            self._p.resetBasePositionAndOrientation(self.obj_dict[key], [x, y, z], quat)
        self._step(240)

    def init_objects(self):
        self.num_objects = np.random.randint(self.min_objects, self.max_objects+1)
        for i in range(self.num_objects):
            obj_type = np.random.choice([self._p.GEOM_BOX], p=[1])
            x = np.random.uniform(0.5, 1.0)
            y = np.random.uniform(-0.4, 0.4)
            z = np.random.uniform(0.6, 0.65)
            size = np.random.uniform(0.015, 0.035, (3,)).tolist()
            rotation = np.random.uniform(0, 90, (3,)).tolist()
            # if obj_type == self._p.GEOM_CAPSULE:
            #     rotation = [0, 0, 0]

            if obj_type == self._p.GEOM_BOX:
                # color = [1.0, 0.0, 0.0, 1.0]
                if np.random.rand() < 0.4:
                    size = [np.random.uniform(0., 0.2), np.random.uniform(0.01, 0.015),
                            np.random.uniform(0.015, 0.025)]
                    # color = [0.0, 1.0, 1.0, 1.0]
            self.obj_dict[i] = utils.create_object(p=self._p, obj_type=obj_type, size=size, position=[x, y, z],
                                                   rotation=rotation, color="random", mass=0.1)

    def state(self):
        rgb, depth, seg = utils.get_image(p=self._p, eye_position=[1.2, 0.0, 1.6], target_position=[0.8, 0., 0.4],
                                          up_vector=[0, 0, 1], height=256, width=256)
        return rgb[:, :, :3], depth, seg

    def step(self, from_obj_id, to_obj_id, sleep=False):
        from_pos, from_quat = self._p.getBasePositionAndOrientation(from_obj_id)
        to_pos, to_quat = self._p.getBasePositionAndOrientation(to_obj_id)
        to_pos = to_pos[:2] + (0.75,)
        traj_time = 0.5
        self.agent.set_cartesian_position(from_pos[:2]+(0.75,),
                                          orientation=self._p.getQuaternionFromEuler([np.pi, 0, 0]),
                                          t=traj_time,
                                          traj=True,
                                          sleep=sleep)
        self.agent.set_cartesian_position(from_pos,
                                          orientation=self._p.getQuaternionFromEuler([np.pi, 0, 0]),
                                          t=traj_time,
                                          traj=True,
                                          sleep=sleep)
        self.agent.close_gripper(traj_time, sleep=sleep)
        self.agent.set_cartesian_position(from_pos[:2]+(0.75,),
                                          orientation=self._p.getQuaternionFromEuler([np.pi, 0, 0]),
                                          t=traj_time,
                                          traj=True,
                                          sleep=sleep)
        self.agent.set_cartesian_position(to_pos,
                                          orientation=self._p.getQuaternionFromEuler([np.pi, 0, 0]),
                                          t=traj_time,
                                          traj=True,
                                          sleep=sleep)
        # self.agent._waitsleep(0.5, sleep=sleep)
        before_pose = self.state_obj_poses()
        self.agent.open_gripper(traj_time, sleep=sleep)
        self.init_agent_pose(t=1.0, sleep=sleep)
        after_pose = self.state_obj_poses()
        effect = after_pose - before_pose
        return effect


class BlocksWorld_v2(BlocksWorld):
    def __init__(self, **kwargs):
        self.traj_t = 1.5
        self.locs = {
            0: [0.8, -0.18, 0.41],
            1: [0.8, 0.0, 0.41],
            2: [0.8, 0.18, 0.41],
            3: [0.5, -0.3, 0.41],
            4: [0.5, 0.0, 0.41],
            5: [0.5, 0.3, 0.41]
        }
        self.sizes = [[0.025, 0.025, 0.05], [0.025, 0.2, 0.025]]
        super(BlocksWorld_v2, self).__init__(**kwargs)

    def init_objects(self):
        if self.min_objects == 1:
            self.num_objects = np.random.choice([1, 2, 3], p=[0.1, 0.2, 0.7])
        else:
            self.num_objects = 3
        obj_types = [1, 0, 0]
        R = np.random.permutation(3)
        self.current_obj_locs = [[] for _ in self.locs]
        i = 0
        obj_ids = []
        while i < self.num_objects:
            loc_idx = np.random.randint(3, 6)
            size_idx = obj_types[R[i]]
            if len(self.current_obj_locs[loc_idx]) > 0:
                continue

            position = self.locs[loc_idx][:2] + [0.6]
            size = self.sizes[size_idx]
            self.current_obj_locs[loc_idx].append(0)
            obj_ids.append(utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                               size=size, position=position, rotation=[0, 0, 0],
                                               mass=0.1, color="random"))
            i += 1
        for i, o_id in enumerate(sorted(obj_ids)):
            self.obj_dict[i] = o_id

    def step(self, from_loc, to_loc, sleep=False):
        target_quat = self._p.getQuaternionFromEuler([np.pi, 0, np.pi/2])
        from_pos = self.locs[from_loc]
        from_top_pos = from_pos[:2] + [0.8]
        to_pos = self.locs[to_loc]
        to_top_pos = to_pos[:2] + [0.8]

        before_pose = self.state_obj_poses()
        self.agent.set_cartesian_position(from_top_pos, orientation=target_quat, t=self.traj_t, traj=True, sleep=sleep)
        self.agent.move_in_cartesian(from_pos, orientation=target_quat, t=self.traj_t, sleep=sleep)
        self.agent.close_gripper(self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(from_top_pos, orientation=target_quat, t=self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(to_top_pos, orientation=target_quat, t=self.traj_t, sleep=sleep, ignore_force=True)
        self.agent._waitsleep(0.3, sleep=sleep)
        self.agent.move_in_cartesian(to_pos, orientation=target_quat, t=self.traj_t, sleep=sleep)
        self.agent._waitsleep(0.5, sleep=sleep)
        self.agent.open_gripper(self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(to_top_pos, orientation=target_quat, t=self.traj_t, sleep=sleep)
        self.init_agent_pose(t=1.0, sleep=sleep)
        after_pose = self.state_obj_poses()
        effect = after_pose - before_pose
        if len(self.current_obj_locs[from_loc]) > 0:
            self.current_obj_locs[from_loc].pop()
        self.current_obj_locs[to_loc].append(0)
        return effect

    def sample_random_action(self):
        if np.random.rand() < 0.2:
            # there might be actions that does not pick any objects
            from_idx = np.random.randint(6)
        else:
            from_idx = np.random.choice([i for i in range(len(self.current_obj_locs))
                                         if len(self.current_obj_locs[i]) > 0])
        to_idx = np.random.randint(6)
        return (from_idx, to_idx)


class BlocksWorld_v3(BlocksWorld):
    def __init__(self, **kwargs):
        self.traj_t = 1.5
        self.x_locs = {0: 0.5, 1: 0.750, 2: 1.0}
        self.y_locs = {0: -0.4, 1: -0.2, 2: 0.0, 3: 0.2, 4: 0.4}
        self.sizes = [[0.025, 0.025, 0.05], [0.025, 0.225, 0.025]]
        super(BlocksWorld_v3, self).__init__(**kwargs)

    def init_objects(self):
        self.num_objects = np.random.randint(self.min_objects, self.max_objects+1)
        obj_types = np.random.binomial(1, 0.3, (self.num_objects,)).tolist()
        while sum(obj_types) > 3:
            obj_types = np.random.binomial(1, 0.3, (self.num_objects,)).tolist()
        obj_types = list(reversed(sorted(obj_types)))
        self.current_obj_locs = [[[] for _ in self.y_locs] for _ in self.x_locs]
        i = 0
        obj_ids = []
        while i < self.num_objects:
            size_idx = obj_types[i]
            xidx = np.random.randint(0, len(self.x_locs))
            yidx = np.random.randint(0, len(self.y_locs))
            if (size_idx == 0) and (len(self.current_obj_locs[xidx][yidx]) > 0):
                continue
            if ((size_idx == 1) and
                    ((len(self.current_obj_locs[xidx][yidx]) > 0) or
                     (len(self.current_obj_locs[xidx][max(0, yidx-1)]) > 0) or
                     (len(self.current_obj_locs[xidx][min(4, yidx+1)]) > 0))):
                continue

            position = [self.x_locs[xidx], self.y_locs[yidx], 0.6]
            size = self.sizes[size_idx]
            self.current_obj_locs[xidx][yidx].append(0)
            if size_idx == 1:
                self.current_obj_locs[xidx][max(0, yidx-1)].append(0)
                self.current_obj_locs[xidx][min(4, yidx+1)].append(0)
            obj_ids.append(utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                               size=size, position=position, rotation=[0, 0, 0],
                                               mass=0.1, color="random"))
            i += 1
        for i, o_id in enumerate(sorted(obj_ids)):
            self.obj_dict[i] = o_id

    def step(self, from_loc, to_loc, sleep=False):
        target_quat = self._p.getQuaternionFromEuler([np.pi, 0, np.pi/2])
        from_pos = [self.x_locs[from_loc[0]], self.y_locs[from_loc[1]], 0.41]
        from_top_pos = from_pos[:2] + [1.0]
        to_pos = [self.x_locs[to_loc[0]], self.y_locs[to_loc[1]], 0.41]
        to_top_pos = to_pos[:2] + [1.0]

        before_pose = self.state_obj_poses()
        self.agent.set_cartesian_position(from_top_pos, orientation=target_quat, t=self.traj_t, traj=False, sleep=sleep)
        self.agent.move_in_cartesian(from_pos, orientation=target_quat, t=self.traj_t, sleep=sleep)
        self.agent.close_gripper(self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(from_top_pos, orientation=target_quat, t=self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(to_top_pos, orientation=target_quat, t=self.traj_t, sleep=sleep, ignore_force=True)
        self.agent.move_in_cartesian(to_pos, orientation=target_quat, t=self.traj_t, sleep=sleep)
        self.agent.open_gripper(self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(to_top_pos, orientation=target_quat, t=self.traj_t, sleep=sleep)
        self.init_agent_pose(t=1.0, sleep=sleep)
        after_pose = self.state_obj_poses()
        effect = after_pose - before_pose
        if len(self.current_obj_locs[from_loc[0]][from_loc[1]]) > 0:
            self.current_obj_locs[from_loc[0]][from_loc[1]].pop()
            self.current_obj_locs[to_loc[0]][to_loc[1]].append(0)
        return effect

    def sample_random_action(self):
        if np.random.rand() < 0.0:
            # there might be actions that does not pick any objects
            xidx = np.random.randint(0, len(self.x_locs))
            yidx = np.random.randint(0, len(self.y_locs))
            from_idx = [xidx, yidx]
        else:
            possible_actions = []
            for i in range(len(self.x_locs)):
                for j in range(len(self.y_locs)):
                    if len(self.current_obj_locs[i][j]) > 0:
                        possible_actions.append([i, j])
            r = np.random.randint(0, len(possible_actions))
            from_idx = possible_actions[r]

        to_idx = [np.random.randint(0, len(self.x_locs)), np.random.randint(0, len(self.y_locs))]
        return (from_idx, to_idx)


class BlocksWorld_v4(BlocksWorld):
    def __init__(self, segments=6, x_area=0.8, y_area=0.8, **kwargs):
        self.traj_t = 2

        print(kwargs)
        self.x_init = 0.5
        self.y_init = -0.4

        self.x_area = x_area
        self.y_area = y_area
        self.segments = segments

        self.x_locs = {}
        self.y_locs = {}

        ds = 0.12
        self.del_x = ds 
        self.del_y = ds

        for i in range(segments):
            self.x_locs[i] = self.x_init + self.del_x * i
            self.y_locs[i] = self.y_init + self.del_y * i
        
        for i, y in enumerate([-0.4, -0.28, -0.16, -0.04, 0.08, 0.2]):
            self.y_locs[i] = y
        # TODO: ADD GRAPH
        self.obj_types = {}
        if 'min_objects' not in kwargs:
            kwargs["min_objects"] = 8
        if 'max_objects' not in kwargs:
            kwargs["max_objects"] = 13

        self.current_obj_locs = [[[] for _ in self.y_locs] for _ in self.x_locs]

        single_size = 0.025
        self.sizes = [[single_size, single_size, 0.05],
                      [single_size, 5*single_size, 0.025],
                      [5*single_size, 0.025, single_size]]
        super(BlocksWorld_v4, self).__init__(**kwargs)
        z_line = 4.2499e-01
        line_color = [1,1,0]
        for i in range(segments):
            self._p.addUserDebugLine([self.x_locs[i], self.y_locs[0], z_line],
                    [self.x_locs[i], self.y_locs[segments-1], z_line],
                    lifeTime = 0,
                    lineWidth = 0.25,
                    lineColorRGB = line_color)
            self._p.addUserDebugLine([self.x_locs[0], self.y_locs[i], z_line],
                    [self.x_locs[segments-1], self.y_locs[i], z_line],
                    lifeTime = 0,
                    lineWidth = 0.125,
                    lineColorRGB = line_color)

    def create_object_from_db(self, state_row):
        obj_type = state_row[-1]
        position = [state_row[0], state_row[1], state_row[2]]
        rotation = [state_row[3], state_row[4], state_row[5]]
        if obj_type == 0:
            return
        size = copy.deepcopy(self.sizes[0])
        if obj_type == 4:
            size = self.sizes[1]
        elif obj_type == 5:
            size = self.sizes[2]
        elif obj_type == 3:
            size[2] = self.sizes[2][2]
        # obj type is never 0
        if obj_type == 0:
            o_id = utils.create_object(p=self._p, obj_type=self._p.GEOM_SPHERE,
                                       size=size, position=position, rotation=rotation,
                                       mass=0.1, color="random")
        elif obj_type == 1:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                        size=size, position=position, rotation=rotation,
                                        mass=0.1, color="random"))
        elif obj_type == 2:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_CYLINDER,
                                        size=size, position=position, rotation=rotation,
                                        mass=0.1, color="random"))
        elif obj_type == 3:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                        size=size, position=position, rotation=rotation,
                                        mass=0.1, color="random"))
        elif obj_type == 4:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                        size=size, position=position, rotation=rotation,
                                        mass=0.1, color="random"))
        elif obj_type == 5:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                        size=size, position=position, rotation=rotation,
                                        mass=0.1, color="random"))
        self.obj_dict[len(self.obj_dict)] = o_id
        self.obj_types[o_id] = obj_type
        return o_id
    def create_object(self, obj_type, xidx, yidx):
        """
        Add an object ot the world, without collusions
        return -1 if it is not possible
        return object id if possible
        """
        if (obj_type < 4) and (len(self.current_obj_locs[xidx][yidx]) > 0):
            return -1
        if ((obj_type == 4) and
                ((len(self.current_obj_locs[xidx][yidx]) > 0) or
                 (len(self.current_obj_locs[xidx][max(0, yidx-1)]) > 0) or
                 (len(self.current_obj_locs[xidx][min(3, yidx+1)]) > 0))):
            return -1
        if ((obj_type == 5) and
                ((len(self.current_obj_locs[xidx][yidx]) > 0) or
                 (len(self.current_obj_locs[max(0, xidx-1)][yidx]) > 0) or
                 (len(self.current_obj_locs[min(3, xidx+1)][yidx]) > 0))):
            return -1

        position = [self.x_locs[xidx], self.y_locs[yidx], 0.45]

        size = copy.deepcopy(self.sizes[0])
        if obj_type == 4:
            size = self.sizes[1]
        elif obj_type == 5:
            size = self.sizes[2]
        elif obj_type == 3:
            size[2] = self.sizes[2][2]
        elif obj_type == 2:
            size[1] = 0.05

        self.current_obj_locs[xidx][yidx].append(obj_type)
        if obj_type == 4:
            if yidx > 0:
                self.current_obj_locs[xidx][yidx-1].append(obj_type)
            if yidx < len(self.y_locs) - 1:
                self.current_obj_locs[xidx][yidx+1].append(obj_type)
        if obj_type == 5:
            if xidx > 0:
                self.current_obj_locs[xidx - 1][yidx].append(obj_type)
            if xidx < len(self.x_locs) - 1:
                self.current_obj_locs[xidx + 1][yidx].append(obj_type)
        o_id = -1
        # obj type is never 0
        if obj_type == 0:
            o_id = utils.create_object(p=self._p, obj_type=self._p.GEOM_SPHERE,
                                       size=size, position=position, rotation=[0, 0, 0],
                                       mass=0.1, color="random")
        elif obj_type == 1:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                        size=size, position=position, rotation=[0, 0, 0],
                                        mass=0.1, color="random"))
        elif obj_type == 2:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_CYLINDER,
                                        size=size, position=position, rotation=[0, 0, 0],
                                        mass=0.1, color="random"))
        elif obj_type == 3:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                        size=size, position=position, rotation=[0, 0, 0],
                                        mass=0.1, color="random"))
        elif obj_type == 4:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                        size=size, position=position, rotation=[0, 0, 0],
                                        mass=0.1, color="random"))
        elif obj_type == 5:
            o_id = (utils.create_object(p=self._p, obj_type=self._p.GEOM_BOX,
                                        size=size, position=position, rotation=[0, 0, np.pi],
                                        mass=0.1, color="random"))
        self.obj_dict[len(self.obj_dict)] = o_id
        self.obj_types[o_id] = obj_type
        return o_id
    
    def init_objects(self):
        '''
        obj_tpes index:
            0 : sphere
            1 : box
            2 : cylinder
            3 : short box
            4 : long box
            5 : long box rotated
        '''
        self.obj_types = {}
        self.num_objects = np.random.randint(self.min_objects, self.max_objects+1)

        # self.num_objects = 1
        # obj_types = [4 for i in range(self.num_objects)]
        obj_types = np.random.randint(1, 6, (self.num_objects,)).tolist()
        obj_types = list(reversed(sorted(obj_types)))
        

        self.current_obj_locs = [[[] for _ in self.y_locs] for _ in self.x_locs]
        i = 0
        while i < self.num_objects:
            obj_type = obj_types[i]
            xidx = np.random.randint(0, len(self.x_locs))
            yidx = np.random.randint(0, len(self.y_locs))

            obj_id = self.create_object(obj_type, xidx, yidx)
            if obj_id == -1:
                continue

            i += 1
        self.create_contact_graph()

    def create_contact_graph(self):
        positions, obj_types = self.state_obj_poses_and_types()
        num_objects = len(self.obj_dict)

        clusters = []
        for i in range(num_objects):
            clusters.append(i)

        self.contact_graph = [[0 for i in range(num_objects)] for k in range(num_objects)]
        for i in range(num_objects):
            for k in range(i+1, num_objects):
                if self.contact_graph[i][k] == 1:
                    continue

                contact_points = self._p.getContactPoints(bodyA=self.obj_dict[i], bodyB=self.obj_dict[k])
                if not len(contact_points) == 0:
                    self.contact_graph[i][k] = 1
                    self.contact_graph[k][i] = 1
                    clusters[k] = clusters[i]
        self.clusters = clusters

        return self.contact_graph, self.clusters

    def step(self, from_loc, to_loc, before_grip_rotation=0, after_grip_rotation=0, sleep=False):
        if self.gui == 0:
            sleep = False
        correction_constant = 0.0
        target_quat = self._p.getQuaternionFromEuler([np.pi, 0, np.pi/2])
        from_pos = [self.x_locs[from_loc[0]] + correction_constant, self.y_locs[from_loc[1]], 0.41]
        from_top_pos = from_pos[:2] + [1.0]
        to_pos = [self.x_locs[to_loc[0]] + correction_constant, self.y_locs[to_loc[1]], 0.41]
        to_top_pos = to_pos[:2] + [1.0]

        before_pose, types = self.state_obj_poses_and_types()
        if before_grip_rotation != 0:
            target_quat = self._p.getQuaternionFromEuler([np.pi, 0, 0])
        self.agent.set_cartesian_position(from_top_pos, orientation=target_quat, t=self.traj_t, traj=False, sleep=sleep)
        self.agent.move_in_cartesian(from_pos, orientation=target_quat, t=self.traj_t, sleep=sleep)
        self.agent.close_gripper(self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(from_top_pos, orientation=target_quat, t=self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(to_top_pos, orientation=target_quat, t=self.traj_t, sleep=sleep, ignore_force=True)
        if after_grip_rotation != 0: 
            if target_quat == self._p.getQuaternionFromEuler([np.pi, 0, np.pi/2]):
                target_quat = self._p.getQuaternionFromEuler([np.pi, 0, 0])
            else:
                target_quat = self._p.getQuaternionFromEuler([np.pi, 0, np.pi/2])
        self.agent.move_in_cartesian(to_pos, orientation=target_quat, t=self.traj_t, sleep=sleep)
        self.agent.open_gripper(self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(to_top_pos, orientation=target_quat, t=self.traj_t, sleep=sleep)
        self.init_agent_pose(t=1.5, sleep=sleep)
        after_pose, types = self.state_obj_poses_and_types()
        effect = after_pose - before_pose

        object_id = -1
        if len(self.current_obj_locs[from_loc[0]][from_loc[1]]) > 0:
            object_id = self.current_obj_locs[from_loc[0]][from_loc[1]].pop()
            self.current_obj_locs[to_loc[0]][to_loc[1]].append(object_id)
        self.create_contact_graph()
        return effect, types

    def state_obj_poses_and_types(self):
        N_obj = len(self.obj_dict)
        pose = np.zeros((N_obj, 6), dtype=np.float32)
        obj_types = np.zeros((N_obj), dtype=np.int8)
        for i in range(N_obj):
            position, quaternion = self._p.getBasePositionAndOrientation(self.obj_dict[i])
            pose[i][:3] = position
            pose[i][3:] = self._p.getEulerFromQuaternion(quaternion)
            obj_types[i] = self.obj_types[self.obj_dict[i]]

        return pose, obj_types

    def get_obj_location(self, obj_id):

        position, quaternion = self._p.getBasePositionAndOrientation(self.obj_dict[obj_id])
        rel_x = position[0] - self.x_init
        x_index = rel_x / self.del_x
        x_index = round(x_index)

        rel_y = position[1] - self.y_init
        y_index = round(rel_y / self.del_y)

        x_index = min(x_index, self.segments - 1)
        y_index = min(y_index, self.segments - 1)
        x_index = max(0, x_index)
        y_index = max(0, y_index)

        return [x_index, y_index]

    def state(self):
        return self.state_obj_poses_and_types()

    def sample_random_action(self):
        action_type = np.random.rand()
        to_idx = [np.random.randint(0, len(self.x_locs)), np.random.randint(0, len(self.y_locs))]
        from_idx = [np.random.randint(0, len(self.x_locs)), np.random.randint(0, len(self.y_locs))]
        before_rotation = np.random.randint(0, 2)
        after_rotation = np.random.randint(0, 2)
        
        if action_type < 0.05:
            # there might be actions that does not pick any objects
            # or actions that does not pick long objects from the middle position
            from_idx = self.get_obj_location(np.random.randint(0, self.num_objects))
            # TODO: perform object location change operations instead of this
        else:
            possible_actions = []
            for i in range((self.num_objects)):
                for k in range(i, (self.num_objects)):
                    if self.clusters[k] != self.clusters[i]:
                        possible_actions.append([i, k])

            if len(possible_actions) != 0:
                action = possible_actions[np.random.randint(0, len(possible_actions))]

                obj_1 = action[0]
                obj_2 = action[1]

                from_idx = self.get_obj_location(obj_1)
                to_idx = self.get_obj_location(obj_2)
            
            action_type = np.random.rand()
            
            # if self.obj_types[self.obj_dict[obj_1]] in [4,5]:
            #     action_type = np.random.rand()
            #     if action_type < 0.5:
            #         _, quaternion = self._p.getBasePositionAndOrientation(self.obj_dict[obj_1])
            #         delta = np.random.choice([1,-1])
            #         axis = 1
            #         if np.all(np.array(self._p.getEulerFromQuaternion(quaternion)[-1] )< 0.5):
            #             axis = 0
            #         from_idx[axis] += delta
            #         to_idx[axis] += delta

            #         from_idx = np.clip(from_idx, 0, self.segments-1)
            #         to_idx = np.clip(to_idx, 0, self.segments-1)
                    #not having this adds too much uncertainty
                    # after_rotation = 0
            if self.obj_types[self.obj_dict[obj_2]] in [4,5]:
                action_type = np.random.rand()
                if action_type < 0.5:
                    _, quaternion = self._p.getBasePositionAndOrientation(self.obj_dict[obj_2])
                    if np.all(np.array(self._p.getEulerFromQuaternion(quaternion)[-1] )< 0.5):
                        to_idx[1] += np.random.choice([1 ,-1])
                    else:
                        to_idx[0] += np.random.choice([1,-1])
                    to_idx = np.clip(to_idx, 0, self.segments-1)


        return (from_idx, to_idx, before_rotation, after_rotation)


class PushEnv(GenericEnv):
    def __init__(self, gui=0, seed=None):
        super(PushEnv, self).__init__(gui=gui, seed=seed)

    def reset(self, seed=None):
        super(PushEnv, self).reset(seed=seed)

        self.obj_dict = {}
        self.init_agent_pose(t=1)
        self.init_objects()
        self._step(40)
        self.agent.close_gripper(1, sleep=True)

    def reset_objects(self):
        for key in self.obj_dict:
            obj_id = self.obj_dict[key]
            self._p.removeBody(obj_id)
        self.obj_dict = {}
        self.init_objects()
        self._step(240)

    def init_objects(self):
        obj_type = np.random.choice([self._p.GEOM_BOX, self._p.GEOM_SPHERE, self._p.GEOM_CYLINDER], p=[0.6, 0.1, 0.3])
        position = [0.8, 0.0, 0.6]
        rotation = [0, 0, 0]
        if self._p.GEOM_CYLINDER:
            r = np.random.uniform(0.05, 0.15)
            h = np.random.uniform(0.05, 0.15)
            size = [r, h]
        else:
            r = np.random.uniform(0.05, 0.15)
            size = [r, r, r]
        size = np.random.uniform(0.015, 0.035, (3,)).tolist()

        self.obj_dict[0] = utils.create_object(p=self._p, obj_type=obj_type, size=size, position=position,
                                               rotation=rotation, color="random", mass=0.1)

    def state(self):
        rgb, depth, seg = utils.get_image(p=self._p, eye_position=[1.5, 0.0, 1.5], target_position=[0.9, 0., 0.4],
                                          up_vector=[0, 0, 1], height=256, width=256)
        return rgb[:, :, :3], depth, seg

    def step(self, action, sleep=False):
        traj_time = 1

        if action == 0:
            self.agent.set_cartesian_position([0.8, -0.15, 0.75],
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
            self.agent.set_cartesian_position([0.8, -0.15, 0.42],
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
            self.agent.move_in_cartesian([0.8, 0.15, 0.42],
                                         self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=traj_time, sleep=sleep)
            self.init_agent_pose(t=0.25, sleep=sleep)
        elif action == 1:
            self.agent.set_cartesian_position([0.95, 0.0, 0.75],
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
            self.agent.set_cartesian_position([0.95, 0.0, 0.42],
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
            self.agent.move_in_cartesian([0.65, 0.0, 0.42],
                                         self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=traj_time, sleep=sleep)
            self.init_agent_pose(t=0.25, sleep=sleep)
        elif action == 2:
            self.agent.set_cartesian_position([0.8, 0.15, 0.75],
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
            self.agent.set_cartesian_position([0.8, 0.15, 0.42],
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
            self.agent.move_in_cartesian([0.8, -0.15, 0.42],
                                         self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=traj_time, sleep=sleep)
            self.init_agent_pose(t=0.25, sleep=sleep)
        elif action == 3:
            self.agent.set_cartesian_position([0.65, 0.0, 0.75],
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
            self.agent.set_cartesian_position([0.65, 0.0, 0.42],
                                              self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=0.5, sleep=sleep)
            self.agent.move_in_cartesian([0.95, 0.0, 0.42],
                                         self._p.getQuaternionFromEuler([np.pi, 0, 0]), t=traj_time, sleep=sleep)
            self.init_agent_pose(t=0.25, sleep=sleep)
