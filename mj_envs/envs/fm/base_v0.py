from time import process_time
import gym
import numpy as np
from mj_envs.envs import env_base
# from mj_envs.utils.xml_utils import reassign_parent
import os
import collections

class DManusBase(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'pose_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": -1.0,
        "bonus": 4.0,
        "penalty": -50,
    }

    def __init__(self, model_path, config_path, target_pose, use_mags=False, **kwargs):

        curr_dir = os.path.dirname(os.path.abspath(__file__))

        self.target_pose = target_pose
        self.use_mags = use_mags
        if use_mags:
            self.DEFAULT_OBS_KEYS.append('mag')

        env_base.MujocoEnv.__init__(self,
                                curr_dir+model_path)
        

        env_base.MujocoEnv._setup(self, 
            obs_keys=self.DEFAULT_OBS_KEYS,
            weighted_reward_keys = self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
            reward_mode = "dense",             
            frame_skip = 40,
            normalize_act = True,
            # obs_range = (-10, 10),
            # seed = None,
            act_mode = "pos",
            is_hardware = False,
            config_path = config_path,
            # rwd_viz = False,
            robot_name = 'dmanus', 
        )


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
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
    
    def get_mag_obs(self, sim):
        # Get contact points and compute resulting magnetic fields
        print([(i,c.geom1,c.geom2) for i,c in enumerate(sim.data.contact)])
        print(sim.data.contact[5].pos)
        return np.zeros((15,))

class FMBase(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'pose_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": -1.0,
        "bonus": 4.0,
        "penalty": -50,
    }
    def __init__(self, model_path, **kwargs):

        curr_dir = os.path.dirname(os.path.abspath(__file__))

        # Process model to ues DManus as end effector
        raw_sim = env_base.get_sim(model_path=curr_dir+model_path)
        raw_xml = raw_sim.model.get_xml()
        processed_xml = reassign_parent(xml_str=raw_xml, receiver_node="panda0_link7", donor_node="DManus_mount")
        processed_model_path = curr_dir+model_path[:-4]+"_processed.xml"
        with open(processed_model_path, 'w') as file:
            file.write(processed_xml)

        # self.sim = env_base.get_sim(model_path=processed_model_path)
        # self.sim_obsd = env_base.get_sim(model_path=processed_model_path)
        # os.remove(processed_model_path)
        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=processed_model_path)
        os.remove(processed_model_path)

        self._setup(**kwargs)


    def _setup(self, 
               target_pose, 
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs):

        self.target_pose = target_pose

        # get env
        # env_base.MujocoEnv.__init__(self,
        #                         # sim = self.sim,
        #                         # sim_obsd = self.sim_obsd,
        #                         frame_skip = 40,
        #                         config_path = config_path,
        #                         obs_keys = self.DEFAULT_OBS_KEYS,
        #                         rwd_keys_wt = self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
        #                         rwd_mode = "dense",
        #                         act_mode = "pos",
        #                         act_normalized = True,
        #                         is_hardware = False,
        #                         **kwargs)
        
        env_base.MujocoEnv.__init__(self,
                                processed_model_path)
        
        os.remove(processed_model_path)

        env_base.MujocoEnv._setup(self, 
            obs_keys=self.DEFAULT_OBS_KEYS,
            weighted_reward_keys = self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
            reward_mode = "dense",             
            frame_skip = 40,
            normalize_act = True,
            # obs_range = (-10, 10),
            # seed = None,
            act_mode = "pos",
            is_hardware = False,
            config_path = config_path,
            # rwd_viz = False,
            robot_name = 'fm', 
        )


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['pose_err'] = obs_dict['qp'] - self.target_pose
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

class DManusEnvFixed(DManusBase):

    def reset(self):
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs


class FMReachEnvFixed(FMBase):

    def reset(self):
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs


class FMReachEnvRandom(FMBase):

    def reset(self):
        self.target_pose = self.np_random.uniform(low=self.sim.model.actuator_ctrlrange[:,0], high=self.sim.model.actuator_ctrlrange[:,1])
        obs = super().reset(self.init_qpos, self.init_qvel)
        return obs
