from src.env.catan_env import CatanEnv
import numpy as np

env = CatanEnv()
obs, info = env.reset()

print("Initial Observation Keys:", obs.keys())
mask = env.get_valid_actions_mask()
valid_indices = np.where(mask == 1)[0]
print(f"Initial Valid Actions count: {len(valid_indices)}")
print(f"Valid Indices: {valid_indices}")

if len(valid_indices) > 0:
    for i in range(5):
        action = valid_indices[0] # Just take the first valid one
        print(f"\n--- Step {i+1} ---")
        print(f"Taking action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Result: Reward={reward}, Terminated={terminated}, Truncated={truncated}")
        
        if terminated or truncated:
            print("Episode ended.")
            break
            
        mask = env.get_valid_actions_mask()
        valid_indices = np.where(mask == 1)[0]
        if len(valid_indices) == 0:
            print("No more valid actions!")
            break
else:
    print("No valid actions found at start!")
