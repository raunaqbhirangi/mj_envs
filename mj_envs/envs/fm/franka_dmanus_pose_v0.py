""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import gym
import numpy as np
from mj_envs.envs import env_base
from mj_envs.utils.xml_utils import reassign_parent, remove_joint
import os
import collections
import torch


class FrankaDmanusPose(env_base.MujocoEnv):
    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'pose_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": -1.0,
        "bonus": 4.0,
        "penalty": -50,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        curr_dir = os.path.dirname(os.path.abspath(__file__))

        # Process model to use DManus as end effector
        raw_sim = env_base.get_sim(model_path=curr_dir + model_path)
        raw_xml = raw_sim.model.get_xml()
        processed_xml = reassign_parent(xml_str=raw_xml, receiver_node="panda0_link7", donor_node="DManus_mount")
        processed_model_path = curr_dir + model_path[:-4] + "_processed.xml"
        with open(processed_model_path, 'w') as file:
            file.write(processed_xml)

        # Process model to use DManus as end effector
        if obsd_model_path == model_path:
            processed_obsd_model_path = processed_model_path
        elif obsd_model_path:
            raw_sim = env_base.get_sim(model_path=curr_dir + obsd_model_path)
            raw_xml = raw_sim.model.get_xml()
            processed_xml = reassign_parent(xml_str=raw_xml, receiver_node="panda0_link7", donor_node="DManus_mount")
            processed_obsd_model_path = curr_dir + obsd_model_path[:-4] + "_processed.xml"
            with open(processed_obsd_model_path, 'w') as file:
                file.write(processed_xml)
        else:
            processed_obsd_model_path = None

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=processed_model_path, obsd_model_path=processed_obsd_model_path, seed=seed)
        os.remove(processed_model_path)
        if processed_obsd_model_path and processed_obsd_model_path != processed_model_path:
            os.remove(processed_obsd_model_path)

        self._setup(**kwargs)

    def _setup(self,
               target_pose,
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs):
        if isinstance(target_pose, np.ndarray):
            self.target_type = 'fixed'
            self.target_pose = target_pose
        elif target_pose == 'random':
            self.target_type = 'random'
            self.target_pose = self.sim.data.qpos.copy()  # fake target for setup

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       frame_skip=40,
                       **kwargs)

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['pose_err'] = obs_dict['qp'] - self.target_pose
        return obs_dict

    def get_reward_dict(self, obs_dict):
        pose_dist = np.linalg.norm(obs_dict['pose_err'], axis=-1)
        far_th = 10

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pose', pose_dist),
            ('bonus', (pose_dist < 1) + (pose_dist < 2)),
            ('penalty', (pose_dist > far_th)),
            # Must keys
            ('sparse', -1.0 * pose_dist),
            ('solved', pose_dist < .5),
            ('done', pose_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def get_target_pose(self):
        if self.target_type == 'fixed':
            return self.target_pose
        elif self.target_type == 'random':
            return self.np_random.uniform(low=self.sim.model.actuator_ctrlrange[:, 0],
                                          high=self.sim.model.actuator_ctrlrange[:, 1])

    def reset(self):
        self.target_pose = self.get_target_pose()
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs


class FrankaDmanusPoseWithBall(FrankaDmanusPose):
    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'target_err', 'ball', 'dball'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        # "pose": -1.0,
        "bonus": 4.0,
        "penalty": -50,
    }

    def __init__(self, model_path, removed_joint_ids, target_xy_range, ball_xy_range, obsd_model_path=None, seed=None,
                 **kwargs):

        self.target_xy_range = target_xy_range
        self.ball_xy_range = ball_xy_range
        self.removed_joint_ids = removed_joint_ids

        self.franka_njoints = 7
        self.dmanus_njoints = 9
        self.njoints = self.franka_njoints + self.dmanus_njoints
        # Add code here to move sites onto palm
        super().__init__(model_path, obsd_model_path, seed, **kwargs)

    def _setup(self,
               target_ball_pos,
               franka_init=None,
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs
               ):

        self.target_sid = self.sim.model.site_name2id('target')
        self.init_sid = self.sim.model.site_name2id('init_ball')

        self.palm_id = self.sim.model.site_bodyid[self.init_sid]

        if isinstance(target_ball_pos, np.ndarray):
            self.target_type = 'fixed'
            self.target_ball_pos = target_ball_pos
        elif target_ball_pos == 'random':
            self.target_type = 'random'
            self.target_ball_pos = self.sim.data.qpos[16:19].copy()  # fake target for setup

        # Make changes for new action and observation spaces
        self.qpos_mask = np.ones_like(self.sim.data.qpos.ravel(), dtype=bool)
        self.qpos_mask[self.removed_joint_ids] = 0

        self.qvel_mask = np.ones_like(self.sim.data.qvel.ravel(), dtype=bool)
        self.qvel_mask[self.removed_joint_ids] = 0

        env_base.MujocoEnv._setup(self, obs_keys=obs_keys,
                                  weighted_reward_keys=weighted_reward_keys,
                                  frame_skip=40,
                                  **kwargs)
        if franka_init is not None:
            self.init_qpos[:7] = franka_init

        self.action_mask = np.ones_like(self.action_space.low, dtype=bool)
        self.action_mask[self.removed_joint_ids] = 0
        act_space_low = self.action_space.low[self.action_mask]
        act_space_high = self.action_space.high[self.action_mask]
        self.action_space = gym.spaces.Box(act_space_low, act_space_high, dtype=np.float32)

        init_robot_qpos = self.init_qpos[:self.franka_njoints + self.dmanus_njoints]
        self.init_robot_action = self.robot.normalize_actions(init_robot_qpos)

    def get_mag_obs(self, sim):
        """
        Use simulation data to identify contacts. Use contact location and force
        to simulate magnetic field data from ReSkin.

        Parameters
        ----------
        sim: MjSim object
        """

        num_mags = len(self.mag_names)
        mag_obs = np.zeros((num_mags, 3))

        # Create a copy of original colors
        colors = self.site_colors.copy()

        # Detection threshold distance for magnetometer
        # (16 mm Manhattan distance)
        mag_thres = 0.035

        # Check for contacts
        for con_id in range(sim.data.ncon):
            c = sim.data.contact[con_id]

            # Get ids for the colliding bodies
            body1_id = sim.model.geom_bodyid[c.geom1]
            body2_id = sim.model.geom_bodyid[c.geom2]

            body2_xmat = sim.data.body_xmat[body2_id].reshape((3, 3))
            # For now just assumes that the second geom is on the hand.
            # TODO: Ask vikash how to check if it belongs to dmanus.
            # ipdb.set_trace()

            # Filter sites with body id matching collision body and make
            # sure they're magnetometers
            sites_mask = (sim.model.site_bodyid == body2_id) * self.mag_mask

            # Find site closest to contact point, if there are any
            # sites on the body
            if np.sum(sites_mask) != 0:
                sites_r = c.pos - sim.data.site_xpos[sites_mask]
                sites_r_local = sites_r @ body2_xmat

                sites_dist = np.linalg.norm(sites_r_local[..., ::2], ord=2, axis=-1)
                # sites_mandist = np.linalg.norm(sites_r, ord=1, axis=-1)

                # v0.1: Just light up the closest magnetometer
                closest_site = np.argmin(sites_dist)
                closest_site_id = np.arange(len(sites_mask))[sites_mask][closest_site]

                # # Change color to red
                # colors[closest_site_id] = [1.,0.,0.,1.]

                # v1.0: Predict magnetic field for all magnetometers within
                # range
                mags_in_range = sites_dist < mag_thres
                mags_in_range_ids = np.arange(len(sites_mask))[sites_mask][mags_in_range]
                # Need to project sites_r to mag coordinate frame
                # ipdb.set_trace()

                contact_location_xy = sites_r_local[..., ::2]
                mag_output = self.get_mag_output(contact_location_xy, mags_in_range, mag_thres)

                mag_obs[sites_mask[self.mag_mask]] = mag_output

                # Color magnetometers based on outputs
                mags_in_range_ids = np.arange(len(sites_mask))[sites_mask][mags_in_range]
                colors[mags_in_range_ids, :3] = np.clip(0.5 + mag_output[mags_in_range] / 2, -1., 1.)

        self.sim.model.site_rgba[:, :] = colors

        return mag_obs.reshape((-1,))

    def get_mag_output(self, contact_location_xy, mags_in_range, mag_thres):
        # Default scaling for input distances is 8mm. Magnetometers here are much further apart.
        # So we use a different scaling factor to make sure NN inputs stay within distribution.
        curr_scaling_mm = self.mag_model_config['input_std'] * 0.001
        new_scaling = mag_thres / np.sqrt(2)
        magout = np.zeros((len(contact_location_xy), 3))
        if np.sum(mags_in_range) > 0:
            scaled_loc = torch.tensor(contact_location_xy / new_scaling, dtype=torch.float)

            with torch.no_grad():
                magdist = self.mag_model(scaled_loc)
                magdist = magdist[mags_in_range]
            magout[mags_in_range] = torch.distributions.normal.Normal(
                magdist[..., :3], torch.exp(magdist[..., 3:])).sample()
            # magout[mags_in_range] = magdist[...,:3]

        return magout

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()

        obs_dict['reduced_qp'] = obs_dict['qp'][self.qpos_mask]
        obs_dict['reduced_qv'] = obs_dict['qv'][self.qvel_mask]

        # Add magnetometer measurements here
        obs_dict['ball'] = self.sim.data.qpos[self.njoints:self.njoints + 3].copy()
        obs_dict['dball'] = self.sim.data.qvel[self.njoints:self.njoints + 3].copy() * self.dt
        obs_dict['target'] = self.sim.data.site_xpos[self.target_sid].copy()

        # Change this to only look at target pose for the ball
        obs_dict['target_err'] = obs_dict['ball'] - self.target_ball_pos
        # print(len(self.sim.data.contact))
        # Add code for mags and magdiffs
        return obs_dict

    def get_reward_dict(self, obs_dict):
        # TODO: Add rewards for ball balancing here
        reach_dist = np.linalg.norm(obs_dict['target_err'], axis=-1)
        far_th = 0.25
        # print(f'Reach dist: {reach_dist}, far_th: {far_th}')
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach', reach_dist),
            ('bonus', (reach_dist < .035) + (reach_dist < .07)),
            ('penalty', (reach_dist > far_th)),
            # Must keys
            ('sparse', -1.0 * reach_dist),
            ('solved', reach_dist < .03),
            ('done', reach_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def reset(self):
        self.sim.data.qpos[:] = self.init_qpos.copy()
        self.sim.forward()

        # Reset init and target sites
        self.sim.model.site_pos[self.target_sid][::2] = self.np_random.uniform(
            low=self.target_xy_range[0], high=self.target_xy_range[1])
        self.sim.model.site_pos[self.init_sid][::2] = self.np_random.uniform(
            low=self.ball_xy_range[0], high=self.ball_xy_range[1])
        self.sim.forward()
        self.target_ball_pos = self.sim.data.site_xpos[self.target_sid].copy()

        # move the ball to the init_site
        qp = self.init_qpos.copy()

        qp[16:19] = self.sim.data.site_xpos[self.init_sid]

        obs = env_base.MujocoEnv.reset(self, reset_qpos=qp)
        return obs

    def step(self, a):
        """
        Step the simulation forward (t => t+1)
        Uses robot interface to safely step the forward respecting pos/ vel limits
        """
        if not hasattr(self, 'action_mask'):
            return super().step(a)
        a = np.clip(a, self.action_space.low, self.action_space.high)
        extended_a = self.init_robot_action.copy()
        extended_a[self.action_mask] = a
        self.last_ctrl = self.robot.step(ctrl_desired=extended_a,
                                         ctrl_normalized=self.normalize_act,
                                         step_duration=self.dt,
                                         realTimeSim=self.mujoco_render_frames,
                                         render_cbk=self.mj_render if self.mujoco_render_frames else None)

        # observation
        obs = self.get_obs()

        # rewards
        self.expand_dims(self.obs_dict)  # required for vectorized rewards calculations
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.squeeze_dims(self.rwd_dict)
        self.squeeze_dims(self.obs_dict)

        # finalize step
        env_info = self.get_env_infos()

        # returns obs(t+1), rew(t), done(t), info(t+1)
        return obs, env_info['rwd_' + self.rwd_mode], bool(env_info['done']), env_info
