import gym
import stable_baselines3
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.ppo.ppo_rnd import RND
from stable_baselines3.ppo.ppo_ospoe_cont import OSPOEContinuous as OSPOE
from stable_baselines3.ppo.ppo_pcpg import PCPG
from stable_baselines3.ppo.ppo_rnd_test import RNDTest
from stable_baselines3.ppo.ppo_rnd_test2 import RNDTest2
from stable_baselines3.common.env_util import make_atari_env, make_vec_env, sparse_reward_wrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

import argparse

import torch

import copy

def get_agent(agent_name, env, n_envs, env_type):
    # hopper setting (resolve hopper)
    # gamma == 0.99 int_gamma == 0.95 ent_coeff == 0 (current, not working)
    # gamma == 0.999 int_gamma == 0.99 ent_coeff == 0 (should be working)
    # gamma == 0.99 int_gamma == 0.99 ent_coeff == 0 (current, not working)
    
    # setting 1 : int_gamma == 0.99 ent_coeff == 0
    # setting 2 : int_gamma == 0.95 ent_coeff == 0 (current, not working)
    # setting 3 : int_gamma == 0.99 ent_coeff == 0.001
    # setting 4 : int_gamma == 0.95 ent_coeff == 0.001
    
    # hopper : 1 env, 2048 steps ent: 0 (OSPOE 1 env, 4096 steps)
    # walker2d : 8 env, 256 steps, (OSPOE 16 env, 256 steps) ent : 0
    # halfcheetah : 8 env, 256 steps, (OSPOE 16 env, 256 steps) ent : 0
    batch_size = 64
    n_steps = 2048 // n_envs
    learning_rate = 3e-4
    n_epochs = 10
    clip_range = 0.2
    if env_type == 'Hopper':
        gamma = 0.999
        int_gamma = 0.99
    else:
        gamma = 0.99
        int_gamma = 0.98
    ent_coef = 0.0
    
    if agent_name == 'PPO':
        # use reported best parameters by stable-baselines zoo
        model = PPO(
            env=env,
            policy='MlpPolicy',
            learning_rate=3e-4,
            vf_coef=0.5,
            ent_coef=0.0,
            tensorboard_log='./mujoco_bk',
            clip_range=0.2,
            n_steps=2048,
            n_epochs=10,
            batch_size=64,
            verbose=1,
        )
    elif agent_name == 'PPOControl':
        # use reported best parameters by stable-baselines zoo
        model = PPO(
            env=env,
            policy='MlpPolicy',
            learning_rate=3e-4,
            vf_coef=0.5,
            ent_coef=ent_coef,
            gamma=gamma,
            tensorboard_log='./mujoco_bk',
            clip_range=0.2,
            n_steps=n_steps,
            n_epochs=10,
            batch_size=64,
            verbose=1,
        )
    elif agent_name == 'ENIAC':
        model = OSPOE(
            env=env,
            policy='MlpPolicy',
            learning_rate=learning_rate,
            vf_coef=0.5,
            ent_coef=ent_coef,
            gamma=gamma,
            int_gamma=int_gamma,
            tensorboard_log='./mujoco_bk',
            clip_range=clip_range,
            n_steps=n_steps,
            n_epochs=n_epochs,
            batch_size=batch_size,
            verbose=1,
        )
    elif agent_name == 'ENIACExtraBonus':
        model = OSPOE(
            env=env,
            policy='MlpPolicy',
            learning_rate=learning_rate,
            vf_coef=0.5,
            ent_coef=ent_coef,
            gamma=gamma,
            int_gamma=int_gamma,
            extra_bonus=True,
            tensorboard_log='./mujoco_bk',
            clip_range=clip_range,
            n_steps=n_steps,
            n_epochs=n_epochs,
            batch_size=batch_size,
            verbose=1,
        )
    elif agent_name == 'OSPOE':
        model = OSPOE(
            env=env,
            policy='MlpPolicy',
            learning_rate=learning_rate,
            vf_coef=0.5,
            ent_coef=ent_coef,
            gamma=gamma,
            int_gamma=int_gamma,
            tensorboard_log='./mujoco_bk',
            clip_range=clip_range,
            n_steps=n_steps * 2,
            n_epochs=n_epochs,
            batch_size=batch_size,
            verbose=1,
        )
    elif agent_name == 'OSPOESparse':
        model = OSPOE(
            env=env,
            policy='MlpPolicy',
            learning_rate=learning_rate,
            vf_coef=0.5,
            ent_coef=ent_coef,
            gamma=gamma,
            int_gamma=int_gamma,
            sparse_sampling=True,
            tensorboard_log='./mujoco_bk',
            clip_range=clip_range,
            n_steps=n_steps * 2,
            n_epochs=n_epochs,
            batch_size=batch_size,
            verbose=1,
        )
    elif agent_name == 'OSPOESparseControl':
        model = OSPOE(
            env=env,
            policy='MlpPolicy',
            learning_rate=learning_rate,
            vf_coef=0.5,
            ent_coef=ent_coef,
            gamma=gamma,
            int_gamma=int_gamma,
            sparse_sampling=True,
            tensorboard_log='./mujoco_bk',
            clip_range=clip_range,
            n_steps=n_steps,
            n_epochs=n_epochs,
            batch_size=batch_size,
            verbose=1,
        )
    elif agent_name == 'OSPOESparseControlExtraBonus':
        model = OSPOE(
            env=env,
            policy='MlpPolicy',
            learning_rate=learning_rate,
            vf_coef=0.5,
            ent_coef=ent_coef,
            gamma=gamma,
            int_gamma=int_gamma,
            sparse_sampling=True,
            extra_bonus=True,
            tensorboard_log='./mujoco_bk',
            clip_range=clip_range,
            n_steps=n_steps,
            n_epochs=n_epochs,
            batch_size=batch_size,
            verbose=1,
        )
    elif agent_name == 'RND':
        model = RND(
            env=env,
            policy='MlpPolicy',
            learning_rate=learning_rate,
            vf_coef=0.5,
            ent_coef=ent_coef,
            gamma=gamma,
            int_gamma=int_gamma,
            tensorboard_log='./mujoco_bk',
            clip_range=clip_range,
            n_steps=n_steps,
            n_epochs=n_epochs,
            batch_size=batch_size,
            verbose=1,
        )
    elif agent_name == 'PCPG':
        model = PCPG(
            env=env,
            policy='MlpPolicy',
            learning_rate=learning_rate,
            vf_coef=0.5,
            ent_coef=ent_coef,
            gamma=gamma,
            int_gamma=int_gamma,
            tensorboard_log='./mujoco_bk',
            clip_range=clip_range,
            n_steps=n_steps,
            n_epochs=n_epochs,
            batch_size=batch_size,
            verbose=1,
        )
    else:
        raise NotImplemented
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="OSPOE")
    parser.add_argument("--env_type", default="OSPOE")
    args = parser.parse_args()

    name = args.name
    # assert name in ['PPO', 'PPOControl', 'RND', 'OSPOE', 'ENIAC', 'PCPG', 'RNDTest', 'RNDTest2', 'OSPOESparse', 'OSPOESparseControl']

    env_type = args.env_type
    env_name = f'{env_type}-v3'
    
    runs = 10
    
    
    if env_type == 'Hopper':
        n_envs = 1
    elif env_type == 'Walker2d' or 'HalfCheetah' or 'HumanoidStandup':
        n_envs = 8
    
    # n_envs = 8
    
    for run in range(runs):
        import torch as th
        # seed the RNG for all devices (both CPU and CUDA)
        th.manual_seed(run+123)

        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
        
        import random
        import numpy as np
        random.seed(run+123)
        np.random.seed(run+123)
        
        if name == 'PPO':
            env = make_vec_env(
                env_name,
                n_envs=1,
                seed=run+123,
                vec_env_cls=DummyVecEnv,
            )
        else:
            env = make_vec_env(
                env_name,
                n_envs=n_envs,
                seed=run+123,
                vec_env_cls=SubprocVecEnv,
            )
        env = VecNormalize(env, norm_reward=False, clip_reward=100)
        
        eval_env = make_vec_env(
            env_name,
            n_envs=5,
            seed=run+123,
            vec_env_cls=SubprocVecEnv,
        )

        # eval_env = make_vec_env(
        #     env_name,
        #     n_envs=1,
        #     seed=run,
        #     vec_env_cls=DummyVecEnv,
        # )
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False, clip_reward=100)
        
        model = get_agent(name, env, n_envs, env_type)
        
        model.learn(
            total_timesteps=int(1e6),
            tb_log_name=f'{name}-{n_envs}-Mujoco_bk_{env_type.lower()}',
            reset_num_timesteps=True,
            # eval parameters
            evaluation=True,
            eval_interval=int(1e4),
            eval_env=eval_env,
            eval_episodes=5,
            target_folder=f'/home/ywang3/workplace/width/RL-Package/Mujoco_eval_{env_type.lower()}',
            filename=f'{name}-seed{run}'
        )