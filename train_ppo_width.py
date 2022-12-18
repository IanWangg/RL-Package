import gym
import stable_baselines3
from stable_baselines3.ppo.ppo_width import PPOWidthDiscrete
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv, VecEnv

if __name__ == '__main__':
    env = make_atari_env(
        'MontezumaRevengeNoFrameskip-v4', 
        wrapper_kwargs={'montezuma': True}, 
        n_envs=128, 
        seed=0,
        vec_env_cls=SubprocVecEnv
    )
    env = VecFrameStack(env, n_stack=4)
    
    model = PPOWidthDiscrete(
        env=env,
        policy='CnnPolicy',
        width='CnnWidth',
        width_query_batch_size=16,
        width_replay_batch_size=256,
        width_learning_rate=1e-4,
        learning_rate=1e-4,
        vf_coef=0.5,
        ent_coef=0.01,
        tensorboard_log='./Exploration',
        n_steps=4,
        n_epochs=4,
        batch_size=256,
        verbose=1,
    )
    
    model.learn(
        total_timesteps=int(1e8),
        tb_log_name=f'PPOWidth_Montezuma_imbalanced_widthtraining',
        reset_num_timesteps=True,
    )
    
    