from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import ale_py
import numpy as np
import torch


from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.logger import configure
import os
import time

N_STACK = 4

DIR_NAME = "collect_data"
if __name__ == '__main__':
    os.makedirs(DIR_NAME, exist_ok=True)
    os.makedirs(os.path.join(DIR_NAME, "monitor"), exist_ok=True)
    env = make_atari_env("ALE/Boxing-v5", n_envs=8
                        ,monitor_dir=os.path.join(DIR_NAME, "monitor"), env_kwargs={"difficulty": 2})
    env = VecFrameStack(env, n_stack=N_STACK)
    
    model = DQN("CnnPolicy", env, verbose=1, device="cuda", buffer_size=100_000)
    
    logger = configure(folder=DIR_NAME, format_strings=["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    for i in range(50):
        model.learn(total_timesteps=200_000, tb_log_name=os.path.join(DIR_NAME), reset_num_timesteps=True)
        model.save(os.path.join(DIR_NAME, "model"))

    env = make_atari_env("ALE/Boxing-v5", n_envs=1
                        ,monitor_dir=os.path.join(DIR_NAME, "monitor"))
    
    env = VecFrameStack(env, n_stack=N_STACK, channels_order='last')
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render('human')
        time.sleep(0.01)
    
