import copy
import pybullet
import pybullet_data
import numpy as np
import random

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
    def __init__(self, segments=6, x_area=0.7, y_area=1.2, **kwargs):
        self.traj_t = 2

        print(kwargs)
        self.x_init = 0.5
        self.y_init = -0.5
        self.x_final = self.x_init + x_area
        self.y_final = self.y_init + y_area

        self.x_area = x_area
        self.y_area = y_area

        ds = 0.1
        self.ds = ds
        self.del_x = ds
        self.del_y = ds

        # for i, y in enumerate([ -0.28, -0.16, -0.04, 0.08, 0.2, 0.32]):
        #     self.y_locs[i] = y
        # TODO: ADD GRAPH
        self.obj_types = {}
        if 'min_objects' not in kwargs:
            kwargs["min_objects"] = 8
        if 'max_objects' not in kwargs:
            kwargs["max_objects"] = 13

        single_size = 0.025
        self.sizes = [[single_size, single_size, 0.05],
                      [single_size, 5*single_size, 0.025],
                      [5*single_size, 0.025, single_size]]
        self.debug_items = []

        super(BlocksWorld_v4, self).__init__(**kwargs)
        self.previous_action = 0

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

    def create_object(self, obj_type, x, y, z=0.5):
        """
        Add an object ot the world, without collusions
        return -1 if it is not possible
        return object id if possible
        """

        position = [x, y, z]

        size = copy.deepcopy(self.sizes[0])
        if obj_type == 4:
            size = self.sizes[1]
        elif obj_type == 5:
            size = self.sizes[2]
        elif obj_type == 1:
            size[2] = self.sizes[2][2]
        elif obj_type == 2:
            size[1] = 0.05

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
            3 : tall
            4 : long box
            5 : long box rotated
        '''
        self.obj_buffer = []
        self.obj_types = {}
        self.num_objects = np.random.randint(self.min_objects, self.max_objects+1)
        # self.num_objects = 1
        # obj_types = [4 for i in range(self.num_objects)]
        obj_types = np.random.randint(1, 6, (self.num_objects - 5,)).tolist()
        obj_types = list(reversed(sorted(obj_types)))
        obj_types = np.concatenate(([i for i in range(1, 6)], obj_types))

        i = 0
        positions = np.array([[0, 0]])
        trials = 0
        while i < self.num_objects:
            obj_type = obj_types[i]
            x = np.random.uniform(0.3, 1.1)
            y = np.random.uniform(-0.7, 0.7)
            z = 0.43
            pos = np.array([[x, y]])
            if np.sqrt(np.sum(pos ** 2)) > 1.2:
                trials += 1
                continue
            if np.any(np.sum(np.abs(positions**2 - pos ** 2), axis=-1) < 2.5 * self.ds):
                trials += 1
                if trials > 10:
                    z = 0.55
                else:
                    continue
            trials = 0
            obj_id = self.create_object(obj_type, x, y, z=z)
            if obj_id == -1:
                continue

            positions = np.concatenate([positions, pos])

            i += 1
        self.cluster_centers = []
        for i in range(np.random.randint(1, 3)):
            self.cluster_centers.append(np.random.randint(0, self.num_objects))
        # TODO: prevent rolling on x y axises
        self.update_contact_graph()

    def update_contact_graph(self):
        return
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

    def get_pos(self):
        return [self.x_locs[self.x_loc], self.y_locs[self.y_loc], 1]

    def get_ground_pos(self):
        pos = self.get_pos()
        pos[2] = 0.41
        return pos

    def remove_grid(self):
        for line in self.debug_items:
            self._p.removeUserDebugItem(line)
        self.debug_items = []

    def print_grid(self, location):
        line_x_color = [0, 0, 1]
        line_y_color = [1, 0, 1]
        x = location[0]
        y = location[1]
        z = location[2]
        for i in [-1, 0, 1]:
            id1 = self._p.addUserDebugLine([x + self.ds * i, y + self.ds, z],
                                           [x + self.ds * i, y - self.ds, z],
                                           lifeTime=0,
                                           lineWidth=0.25,
                                           lineColorRGB=line_x_color)
            id2 = self._p.addUserDebugLine([x + self.ds, y + self.ds * i, z],
                                           [x - self.ds, y + self.ds * i, z],
                                           lifeTime=0,
                                           lineWidth=0.125,
                                           lineColorRGB=line_y_color)
            self.debug_items.append(id1)
            self.debug_items.append(id2)

    def step(self, obj1_id, obj2_id, dx1, dy1, dx2, dy2, grap_angle, put_angle, sleep=False):
        obj1_loc, quat = self._p.getBasePositionAndOrientation(self.obj_dict[obj1_id])
        obj2_loc, _ = self._p.getBasePositionAndOrientation(self.obj_dict[obj2_id])

        # use these if you want to ensure grapping
        # euler_rot = self._p.getEulerFromQuaternion(quat)
        # quat = self._p.getQuaternionFromEuler([np.pi,0.0,euler_rot[0] + np.pi/2])

        approach_angle1 = [np.pi, 0, 0]
        approach_angle2 = [np.pi, 0, 0]
        if grap_angle:
            approach_angle1 = [np.pi, 0, np.pi/2]
        if put_angle:
            approach_angle2 = [np.pi, 0, np.pi/2]
        quat1 = self._p.getQuaternionFromEuler(approach_angle1)
        quat2 = self._p.getQuaternionFromEuler(approach_angle2)

        obj1_loc = list(obj1_loc)
        obj2_loc = list(obj2_loc)
        if sleep:
            self.remove_grid()
            self.print_grid(obj1_loc)
            self.print_grid(obj2_loc)
        obj1_loc[0] += dx1 * self.ds
        obj2_loc[0] += dx2 * self.ds
        obj1_loc[1] += dy1 * self.ds
        obj2_loc[1] += dy2 * self.ds

        up_pos_1 = copy.deepcopy(obj1_loc)
        up_pos_1[2] = 1

        up_pos_2 = copy.deepcopy(obj2_loc)
        up_pos_2[2] = 1
        state1, types = self.state_obj_poses_and_types()
        self.agent.move_in_cartesian(up_pos_1, orientation=quat1, t=self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(obj1_loc, orientation=quat1, t=self.traj_t, sleep=sleep)
        self.agent.close_gripper(sleep=sleep)
        self.agent.move_in_cartesian(up_pos_1, orientation=quat1, t=self.traj_t, sleep=sleep)
        state2, _ = self.state_obj_poses_and_types()
        if approach_angle1 != approach_angle2:
            self.agent.move_in_cartesian(up_pos_1, orientation=quat2, t=self.traj_t, sleep=sleep)
        self.agent.move_in_cartesian(up_pos_2, orientation=quat2, t=self.traj_t*2, sleep=sleep)
        state3, _ = self.state_obj_poses_and_types()
        self.agent.move_in_cartesian(obj2_loc, orientation=quat2, t=self.traj_t, sleep=sleep)
        self.agent.open_gripper()
        self.agent.move_in_cartesian(up_pos_2, orientation=quat2, t=self.traj_t, sleep=sleep)
        state4, _ = self.state_obj_poses_and_types()
        effect = np.concatenate([state2 - state1, state4 - state3], axis=1)

        self.init_agent_pose(1, 0.5)
        return state1, effect, types

    def state_obj_poses_and_types(self):
        N_obj = len(self.obj_dict)
        pose = np.zeros((N_obj, 9), dtype=np.float32)
        obj_types = np.zeros((N_obj), dtype=np.int8)
        for i in range(N_obj):
            position, quaternion = self._p.getBasePositionAndOrientation(self.obj_dict[i])
            pose[i][:3] = position
            euler_angles = self._p.getEulerFromQuaternion(quaternion)
            pose[i][3:] = [np.cos(euler_angles[0]), np.sin(euler_angles[0]),
                           np.cos(euler_angles[1]), np.sin(euler_angles[1]),
                           np.cos(euler_angles[2]), np.sin(euler_angles[2])]
            obj_types[i] = self.obj_types[self.obj_dict[i]]

        return pose, obj_types

    def get_obj_location(self, obj_id):

        position, quaternion = self._p.getBasePositionAndOrientation(self.obj_dict[obj_id])

        return position

    def state(self):
        return self.state_obj_poses_and_types()

    def sample_random_action(self):
        obj1 = np.random.randint(self.num_objects)
        obj2 = np.random.choice(self.cluster_centers)

        while obj1 in self.cluster_centers:
            obj1 = np.random.randint(self.num_objects)
        probs = np.random.rand(8)
        dx1, dy1, dx2, dy2 = 0, 0, 0, 0
        dxdy_pairs = [[0, 0], [0, 1], [1, 0],
                      [-1, 0], [0, -1], [-1, 1],
                      [-1, -1], [1, 1], [1, -1]]
        dxdy1 = np.random.choice(
            np.arange(len(dxdy_pairs)),
            p=[0.5] + [0.1] * 4 + [0.025] * 4
        )
        dxdy2 = np.random.choice(
            np.arange(len(dxdy_pairs)),
            p=[0.4] + [0.125] * 4 + [0.025] * 4
        )

        [dx2, dy2] = dxdy_pairs[dxdy2]
        [dx1, dy1] = dxdy_pairs[dxdy1]

        rot_before = 1 if probs[4] < 0.5 else 0
        rot_after = 1 if probs[5] < 0.5 else 0
        return [obj1, obj2, dx1, dy1, dx2, dy2, rot_before, rot_after]
        # return [obj1, obj2, 0,0,0,0, 0,0]

    def sample_3_objects_moving_together(self):
        long_objects = []
        smalls = []

        for i in self.obj_dict.keys():
            if self.obj_types[self.obj_dict[i]] == 4:
                long_objects.append(i)
            else:
                smalls.append(i)

        long_object = random.choice(long_objects)
        small1, small2 = random.choices(smalls, k=2)

        act = self.sample_random_action()
        if long_object == 4:
            self.obj_buffer.append(
                [small1, long_object, 0, 0, 1, 0, 0, 0]
            )
            self.obj_buffer.append(
                [small2, long_object, 0, 0, -1, 0, 0, 0]
            )
            act[6] = 0
        else:
            self.obj_buffer.append(
                [small1, long_object, 0, 0, 0, 1, 0, 0]
            )
            self.obj_buffer.append(
                [small2, long_object, 0, 0, 0, -1, 0, 0]
            )
            act[6] = 1
        act[2] = 0
        act[3] = 0
        act[0] = long_object

        self.obj_buffer.append(act)
        return self.obj_buffer

    def sample_ungrappable(self):
        tall = 0
        small = 0
        for i in self.obj_dict.keys():
            if self.obj_types[self.obj_dict[i]] == 3:
                tall = i
            elif self.obj_types[self.obj_dict[i]] < 3:
                small = i
        axis = random.random()
        if axis < 0.5:
            dx = np.random.choice([1, -1])
            self.obj_buffer.append([tall, small, 0, 0, dx, 0, 0, 0])
            self.obj_buffer.append([small, np.random.randint(0, self.num_objects), 0, 0, 0, 0, 1, 0])
        else:
            dy = np.random.choice([1, -1])
            self.obj_buffer.append([tall, small, 0, 0, 0, dy, 0, 0])
            self.obj_buffer.append([small, np.random.randint(0, self.num_objects), 0, 0, 0, 0, 0, 0])
        return self.obj_buffer


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
