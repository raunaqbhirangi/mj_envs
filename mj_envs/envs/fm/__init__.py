from gym.envs.registration import register
import numpy as np
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))

# Reach to fixed target
# register(
#     id='DManusReachFixed-v0',
#     entry_point='mj_envs.envs.fm.franka_dmanus_pose_v0:FMReachEnvFixed',
#     max_episode_steps=50, #50steps*40Skip*2ms = 4s
#     kwargs={
#             'model_path': '/assets/dmanus.xml',
#             'config_path': curr_dir+'/assets/dmanus.config',
#             'target_pose': np.array([0, 1, 1, 0, 1, 1, 0, 1, 1]),
#         }
# )

# Pose to fixed target
register(
    id='rpFrankaDmanusPoseFixed-v0',
    entry_point='mj_envs.envs.fm.franka_dmanus_pose_v0:FrankaDmanusPose',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
            'model_path': '/assets/franka_dmanus.xml',
            'config_path': curr_dir+'/assets/franka_dmanus.config',
            'target_pose': np.array([0, 0, 0, -1.57, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1]),
        }
)

# Pose to random target
register(
    id='rpFrankaDmanusPoseRandom-v0',
    entry_point='mj_envs.envs.fm.franka_dmanus_pose_v0:FrankaDmanusPose',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
            'model_path': '/assets/franka_dmanus.xml',
            'config_path': curr_dir+'/assets/franka_dmanus.config',
            'target_pose': 'random'
        }
)

# Ball random target from oracle info
register(
    id='rpFrankaDmanusBallBalanceOracle-v0',
    entry_point='mj_envs.envs.fm.franka_dmanus_pose_v0:FrankaDmanusPoseWithBall',
    max_episode_steps=100, #50steps*40Skip*2ms = 4s
    # Franka init specifies the reset joint angles
    kwargs={
            'model_path': '/assets/franka_dmanus_ball.xml',
            'config_path': os.path.join(curr_dir,'assets/franka_dmanus_ball.config'),
            'removed_joint_ids': [0,1,2,3,5],
            'obs_keys': ['qp', 'qv', 'ball', 'dball'],
            'franka_init': np.array([0.000, 1.5, 1.6, 0.0, -0.0, 0.0, -0.0]),
            'target_xy_range': np.array(([0.0, 0.05], [0.0, 0.05])),
            'ball_xy_range': np.array(([-0.05, 0.0], [0.05, 0.1])),
            'target_ball_pos': 'random',
            # 'mag_model_path': curr_dir+'/reskin_files/model.pt'
        }
)

# Ball to random target from reskin data
register(
    id='rpFrankaDmanusBallBalance-v0',
    entry_point='mj_envs.envs.fm.franka_dmanus_pose_v0:FrankaDmanusPoseWithBall',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
            'model_path': '/assets/franka_dmanus_ball.xml',
            'config_path': os.path.join(curr_dir,'assets/franka_dmanus_ball.config'),
            'obs_keys': ['qp', 'qv', 'mag', 'dmag'],
            'target_xy_range': np.array(([0.0, 0.05], [0.0, 0.05])),
            'ball_xy_range': np.array(([-0.05, 0.0], [0.05, 0.1])),
            'target_ball_pos': 'random',
            # 'mag_model_path': curr_dir+'/reskin_files/model.pt'
        }
)