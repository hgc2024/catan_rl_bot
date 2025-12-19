from src.env.catan_env import CatanEnv
import numpy as np
import sys

env = CatanEnv()
obs, info = env.reset()
mask = env.get_valid_actions_mask()
valid_indices = np.where(mask == 1)[0]

if len(valid_indices) > 0:
    action = valid_indices[0]
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"S1_R: {reward}")
    print(f"S1_T: {terminated}")
    
    if not terminated:
        mask2 = env.get_valid_actions_mask()
        valid_indices2 = np.where(mask2 == 1)[0]
        if len(valid_indices2) > 0:
            action2 = valid_indices2[0]
            obs, reward, terminated, truncated, info = env.step(action2)
            print(f"S2_R: {reward}")
            print(f"S2_T: {terminated}")
        else:
            print("S2_NO_ACTIONS")
else:
    print("NO_INITIAL_ACTIONS")
sys.stdout.flush()
