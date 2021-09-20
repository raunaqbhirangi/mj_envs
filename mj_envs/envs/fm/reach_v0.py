from mj_envs.envs.fm.base_v0 import DManusBase
import numpy as np
import collections

import ipdb

class DManusFixedReach(DManusBase):
    
    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'mag', 'pose_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": -1.0,
        "bonus": 4.0,
        "penalty": -50,
    }
    
    def __init__(self, model_path, config_path, 
        obs_keys, target_pose, **kwargs):
        
        self.target_pose = target_pose
        super().__init__(model_path, config_path, obs_keys, **kwargs)
        

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['pose_err'] = obs_dict['qp'] - self.target_pose
        obs_dict['mag'] = self.get_mag_obs(sim)

        return obs_dict

    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['pose_err'], axis=-1)
        far_th = 10

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   reach_dist),
            ('bonus',   (reach_dist<1) + (reach_dist<2)),
            ('penalty', (reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*reach_dist),
            ('solved',  reach_dist<.5),
            ('done',    reach_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

class DManusBallOnPalmReach(DManusBase):

    DEFAULT_OBS_KEYS = [
        'joints', 'djoints', 'mag'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": -1.0,
        "bonus": 4.0,
        "penalty": -50.,
    }
    
    def __init__(self,
        model_path,
        config_path,
        obs_keys=DEFAULT_OBS_KEYS,
        target_xy_range=np.array(([0.0, 0.0], [0.0, 0.0])),
        ball_xy_range=np.array(([-.05, -.05], [.05, .05])),
        **kwargs):

        self.target_xy_range = target_xy_range
        self.ball_xy_range = ball_xy_range
        self.last_mag = None

        DManusBase.__init__(self, model_path,
        config_path,obs_keys,**kwargs)

    def _setup(self, **kwargs):
        self.target_sid = self.sim.model.site_name2id('target')
        self.init_sid = self.sim.model.site_name2id('init_ball')

        self.palm_id = self.sim.model.site_bodyid[self.init_sid]
        DManusBase._setup(self,**kwargs)

    def get_obs_dict(self, sim):
        # Ensure here that the ball's position and velocity are not (or are)
        # in the qp and qv. Appropriately assign obs_keys.
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        # obs_dict['qp'] = sim.data.qpos.copy()
        # obs_dict['qv'] = sim.data.qvel.copy()
        
        obs_dict['joints'] = self.sim.data.qpos[:11].copy()
        obs_dict['djoints'] = self.sim.data.qpos[:11].copy() * self.dt
        obs_dict['ball'] = self.sim.data.qpos[11:14].copy()
        obs_dict['dball'] = self.sim.data.qvel[11:14].copy() * self.dt
        obs_dict['target'] = self.sim.data.site_xpos[self.target_sid].copy()
        # ipdb.set_trace()
        # Change this to only look at target pose for the ball
        obs_dict['target_err'] = obs_dict['ball'] - obs_dict['target']
        obs_dict['mag'] = self.get_mag_obs(self.sim)

        if self.last_mag is None:
            self.last_mag = obs_dict['mag'].copy()
        obs_dict['magdiff'] = obs_dict['mag'] - self.last_mag

        return obs_dict

    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['target_err'], axis=-1)
        far_th = 0.25
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   reach_dist),
            ('bonus',   (reach_dist<.035) + (reach_dist<.07)),
            ('penalty', (reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*reach_dist),
            ('solved',  reach_dist<.02),
            ('done',    reach_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict
    
    def reset(self):
        # If target and init_pos are in world frame, this works
        # generate targets
        # self.sim.model.site_pos[self.target_sid][:2] = self.np_random.uniform(
        #     low=self.target_xy_range[0], high=self.target_xy_range[1])
        # self.sim.forward()

        # # generate object
        # qp = self.init_qpos.copy()
        # qp[11:13] = self.np_random.uniform(
        #     low=self.ball_xy_range[0], high=self.ball_xy_range[1])
        # self.sim.model.site_pos[self.init_sid][:2] = qp[11:13]
        
        # If target and init_pos are in the palm frame, this should work
        self.sim.model.site_pos[self.target_sid][::2] = self.np_random.uniform(
            low=self.target_xy_range[0], high=self.target_xy_range[1])
        self.sim.model.site_pos[self.init_sid][::2] = self.np_random.uniform(
            low=self.ball_xy_range[0], high=self.ball_xy_range[1])
        self.sim.forward()

        qp = self.init_qpos.copy()

        # body_id = self.sim.model.site_bodyid[self.init_sid]
        # qp[11:14] = self.sim.data.body_xpos[body_id] + self.sim.data.body_xmat[body_id].reshape((3,3)) @ self.sim.model.site_pos[self.init_sid]
        
        qp[11:14] = self.sim.data.site_xpos[self.init_sid]
        obs = super().reset(reset_qpos=qp)
        return obs