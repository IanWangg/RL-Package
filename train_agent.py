import gym
import stable_baselines3
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.ppo.ppo_rnd import RND
from stable_baselines3.ppo.ppo_ospoe_cont import OSPOEContinuous as OSPOE
from stable_baselines3.common.env_util import make_atari_env, make_vec_env, sparse_reward_wrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

def get_agent(agent_name, env, n_envs):
    coeff = int(32 / n_envs)
    if agent_name == 'PPO':
        model = PPO(
            env=env,
            policy='MlpPolicy',
            learning_rate=1e-4,
            vf_coef=0.5,
            ent_coef=0.001,
            tensorboard_log='./MountainCarFinal',
            clip_range=0.1,
            n_steps=16*coeff,
            n_epochs=4,
            batch_size=16,
            verbose=1,
        )
    elif agent_name == 'ENIAC':
        model = OSPOEContinuous(
            env=env,
            policy='MlpPolicy',
            learning_rate=1e-4,
            vf_coef=0.5,
            ent_coef=0.001,
            gamma=0.999,
            tensorboard_log='./MountainCarFinal',
            clip_range=0.1,
            n_steps=16*coeff,
            n_epochs=4,
            batch_size=16,
            verbose=1,
        )
    elif agent_name == 'OSPOE':
        model = OSPOEContinuous(
            env=env,
            policy='MlpPolicy',
            learning_rate=1e-4,
            vf_coef=0.5,
            ent_coef=0.001,
            gamma=0.999,
            tensorboard_log='./MountainCarFinal',
            clip_range=0.1,
            n_steps=32*coeff,
            n_epochs=6,
            batch_size=16,
            verbose=1,
        )
    elif agent_name == 'RND':
        model = RND(
            env=env,
            policy='MlpPolicy',
            learning_rate=1e-4,
            vf_coef=0.5,
            ent_coef=0.001,
            gamma=0.999,
            tensorboard_log='./MountainCarFinal',
            clip_range=0.1,
            n_steps=16*coeff,
            n_epochs=4,
            batch_size=16,
            verbose=1,
        )
    else:
        raise NotImplemented
    
    return model

if __name__ == '__main__':
    agent_name = 'PPO'
    runs = 
    n_envs = 1
    assert agent_name in ['PPO', 'RND', 'OSPOE', 'ENIAC']
    
    env_name = 'MountainCarContinuous-v0'
    
    for name in ['PPO', 'RND', 'OSPOE', 'ENIAC']:
        for run in range(runs):
            env = make_vec_env(
                env_name,
                n_envs=n_envs,
                seed=run,
                vec_env_cls=DummyVecEnv,
            )
            env = VecNormalize(env, norm_reward=False, clip_reward=100)
            
            model = get_agent(agent_name, env, n_envs)
            
            model.learn(
                total_timesteps=int(5e4),
                tb_log_name=f'{agent_name}-{n_envs}-MountainCarContinuous',
                reset_num_timesteps=True,
            )