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
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
            
        # Change this to only look at target pose for the ball
        obs_dict['pose_err'] = obs_dict['qp'] - self.target_pose
        if self.use_mags:
            obs_dict['mag'] = self.get_mag_obs(sim)
        return obs_dict
    
    
            