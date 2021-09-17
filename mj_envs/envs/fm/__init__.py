from gym.envs.registration import register
import numpy as np
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))

# Reach to fixed target
# register(
#     id='DManusReachFixed-v0',
#     entry_point='mj_envs.envs.fm.base_v0:FMReachEnvFixed',
#     max_episode_steps=50, #50steps*40Skip*2ms = 4s
#     kwargs={
#             'model_path': '/assets/dmanus.xml',
#             'config_path': curr_dir+'/assets/dmanus.config',
#             'target_pose': np.array([0, 1, 1, 0, 1, 1, 0, 1, 1]),
#         }
# )

# Reach to fixed target
register(
    id='FMReachFixed-v0',
    entry_point='mj_envs.envs.fm.base_v0:FMReachEnvFixed',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
            'model_path': '/assets/robopen.xml',
            'config_path': curr_dir+'/assets/robopen.config',
            'target_pose': np.array([0, 0, 0, -1.57, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1]),
        }
)

register(
    id='DManusFixedReach-v0',
    entry_point='mj_envs.envs.fm.reach_v0:DManusFixedReach',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
            'model_path': '/assets/dmanus.xml',
            'config_path': curr_dir+'/assets/dmanus.config',
            'target_pose': np.array([0, 1, 1, 0, 1, 1, 0, 1, 1]),
            'obs_keys': ['qp', 'qv', 'mag', 'pose_err']
            # 'mag_model_path': curr_dir+'/reskin_files/model.pt'
        }
)

register(
    id='DManusReachWrist-v0',
    entry_point='mj_envs.envs.fm.base_v0:DManusEnvWrist',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
            'model_path': '/assets/dmanus_wrist/dmanus_wrist.xml',
            'config_path': curr_dir+'/assets/dmanus_wrist/dmanus_wrist.config',
            'target_pose': np.array([0,0,0, 1, 1, 0, 1, 1, 0, 1, 1]),
            # 'mag_model_path': curr_dir+'/reskin_files/model.pt'
        }
)

register(
    id='DManusBallOnPalmReach-v0',
    entry_point='mj_envs.envs.fm.reach_v0:DManusBallOnPalmReach',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
            'model_path': '/assets/dmanus_wrist/dmanus_wrist_with_ball.xml',
            'config_path': curr_dir+'/assets/dmanus_wrist/dmanus_wrist.config',
            'obs_keys': ['joints', 'djoints', 'mag'],
            'target_xy_range': np.array(([0.0, 0.0], [0.0, 0.0])),
            'ball_xy_range': np.array(([-.05, -.05], [.05, .05])),
            # 'mag_model_path': curr_dir+'/reskin_files/model.pt'
        }
)