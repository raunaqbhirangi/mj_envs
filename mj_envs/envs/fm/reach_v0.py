from mj_envs.envs.fm.base_v0 import DManusBase
import numpy as np
import collections

import ipdb

class DManusReachFixed(DManusBase):
    
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

    def __init__(self,
        model_path,
        config_path,
        target_pose,
        use_mags=False,
        **kwargs):

        super().__init__(model_path,
        config_path,
        target_pose,
        use_mags=False,**kwargs)

    def get_obs_dict(self, sim):
        # Ensure here that the ball's position and velocity are not (or are)
        # in the qp and qv. Appropriately assign obs_keys.
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['joints'] = self.data.qpos[:11].copy()
        obs_dict['djoints'] = self.data.qpos[:11].copy()
        obs_dict['ball'] = self.data.qpos[11:14].copy()
        obs_dict['dball'] = self.data.qvel[11:14].copy()
        # ipdb.set_trace()
        # Change this to only look at target pose for the ball
        obs_dict['pose_err'] = obs_dict['qp']
        if self.use_mags:
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