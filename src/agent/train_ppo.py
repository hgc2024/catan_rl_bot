
import gymnasium as gym
import numpy as np
import torch
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.env.catan_env import CatanEnv
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_valid_actions_mask()

def make_env():
    env = CatanEnv()
    env = ActionMasker(env, mask_fn)
    env = Monitor(env)
    return env

if __name__ == "__main__":
    # Hyperparameters
    n_envs = 16
    n_steps = 2048
    batch_size = 1024
    gamma = 0.995
    ent_coef = 0.01
    learning_rate = 3e-4
    total_timesteps = 1_000_000 # Initial run
    
    # Create Vector Env
    vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
    
    # Custom Policy Network (Shared [512, 256])
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256], vf=[512, 256]) 
    )

    # Initialize PPO
    model = MaskablePPO(
        MaskableMultiInputActorCriticPolicy,
        vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"Starting training on {model.device} with {n_envs} envs...")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./logs/', name_prefix='ppo_catan')
    
    try:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        model.save("ppo_catan_final")
        print("Training complete. Model saved.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")
        model.save("ppo_catan_interrupted")
        print("Model saved to 'ppo_catan_interrupted.zip'.")
    finally:
        vec_env.close()
        print("Environment closed.")
