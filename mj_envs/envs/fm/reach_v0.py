from mj_envs.envs.fm.base_v0 import DManusBase
import numpy as np
import collections

class DManusBallOnPalmReachEnvV0(DManusBase):

    def __init__(self,
        model_path,
        config_path,
        target_pose,
        use_mags=False,
        **kwargs):
        super().__init__(**kwargs)

    def get_obs_dict(self, sim):
        # Ensure here that the ball's position and velocity are not (or are)
        # in the qp and qv. Appropriately assign obs_keys.
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
            
        # Change this to only look at target pose for the ball
        obs_dict['pose_err'] = obs_dict['qp'] - self.target_pose
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