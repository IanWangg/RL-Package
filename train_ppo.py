import gym
import stable_baselines3
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.ppo.ppo_rnd import RND
from stable_baselines3.ppo.ppo_ospoe_cont import OSPOEContinuous as OSPOE
from stable_baselines3.common.env_util import make_atari_env, make_vec_env, sparse_reward_wrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

import bsuite
from bsuite.utils import gym_wrapper

if __name__ == '__main__':
    env_name = 'MountainCarContinuous-v0'
    
    '''
    def env_name():
        env = bsuite.load_from_id('deep_sea/0')
        gym_env = gym_wrapper.GymFromDMEnv(env)
        return gym_env
    '''
    
    # ENIAC : 32 env, 16 steps, 4 epoch 
    # OSPOE : 32 env, 32 steps, 6 epoch
    
    env = make_vec_env(
        env_name, 
        # wrapper_class=sparse_reward_wrapper,
        # wrapper_kwargs={'montezuma': False},
        n_envs=1,
        seed=1,
        vec_env_cls=DummyVecEnv
    )
    
    eval_env = make_vec_env(env_name, n_envs=1)
    env = VecNormalize(env, norm_reward=False, clip_reward=100)
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False, clip_reward=100)
    eval_env.obs_rms = env.obs_rms
    
    
    model = PPO(
        env=env,
        policy='MlpPolicy',
        learning_rate=1e-4,
        vf_coef=0.5,
        ent_coef=0.001,
        tensorboard_log='./MountainCarDiscrete',
        clip_range=0.1,
        n_steps=32 * 16,
        n_epochs=4,
        batch_size=16,
        verbose=1,
    )
    
    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=int(5000)
    )
    
    model.learn(
        total_timesteps=int(3e5),
        tb_log_name=f'PPO_singleenv',
        callback=eval_callback,
        reset_num_timesteps=True,
    )