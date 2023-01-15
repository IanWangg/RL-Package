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

def get_agent(agent_name, env, n_envs):
    # hopper : 1 env, 2048 steps
    # walker2d : 32 env, 64 steps, (OSPOE 64 env, 64 steps)
    # halfcheetah : 1 env, 2048 steps (0 ent coeff)
    batch_size = 8
    n_steps = 16
    learning_rate = 1e-4
    n_epochs = 4
    clip_range = 0.2
    gamma = 0.99
    int_gamma = 0.99
    ent_coef = 0
    
    if agent_name == 'PPO':
        # use reported best parameters by stable-baselines zoo
        model = PPO(
            env=env,
            policy='MlpPolicy',
            learning_rate=learning_rate,
            vf_coef=0.5,
            ent_coef=ent_coef,
            gamma=gamma,
            tensorboard_log='./mujoco_bk',
            clip_range=clip_range,
            n_steps=n_steps,
            n_epochs=n_epochs,
            batch_size=batch_size,
            verbose=1,
        )
    elif agent_name == 'PPOControl':
        # use reported best parameters by stable-baselines zoo
        model = PPO(
            env=env,
            policy='MlpPolicy',
            learning_rate=1e-4,
            vf_coef=0.5,
            ent_coef=0.0,
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
            n_steps=n_steps,
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
    elif agent_name == 'RNDTest':
        model = RNDTest(
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
    elif agent_name == 'RNDTest2':
        model = RNDTest2(
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
    else:
        raise NotImplemented
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="OSPOE")
    args = parser.parse_args()

    name = args.name
    runs = 5
    n_envs = 16
    assert name in ['PPO', 'PPOControl', 'RND', 'OSPOE', 'ENIAC', 'PCPG', 'RNDTest', 'RNDTest2']

    env_type = 'MountainCar'
    env_name = f'{env_type}-v0'
    
    
    for run in range(runs):
        for name in ['PPO', 'RND', 'OSPOE', 'ENIAC', 'PCPG']:
            import torch as th
            # seed the RNG for all devices (both CPU and CUDA)
            th.manual_seed(run+100)

            th.backends.cudnn.deterministic = True
            th.backends.cudnn.benchmark = False
            
            import random
            import numpy as np
            random.seed(run+100)
            np.random.seed(run+100)
            
        
            env = make_vec_env(
                env_name,
                n_envs=n_envs,
                seed=run+100,
                vec_env_cls=SubprocVecEnv,
            )
            env = VecNormalize(env, norm_reward=False, clip_reward=100)
            
            eval_env = make_vec_env(
                env_name,
                n_envs=5,
                seed=run,
                vec_env_cls=SubprocVecEnv,
            )
            eval_env = VecNormalize(eval_env, training=False, norm_reward=False, clip_reward=100)
            
            model = get_agent(name, env, n_envs)
            
            model.learn(
                total_timesteps=int(1e5),
                tb_log_name=f'{name}-{n_envs}-{env_type.lower()}',
                reset_num_timesteps=True,
                # eval parameters
                evaluation=True,
                eval_interval=int(1e3),
                eval_env=eval_env,
                eval_episodes=5,
                target_folder=f'/home/ywang3/workplace/width/RL-Package/MountainCarFinalDiscrete_eval_env{n_envs}',
                filename=f'{name}-seed{run}'
            )