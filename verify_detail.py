from src.env.catan_env import CatanEnv
import numpy as np

env = CatanEnv()
obs, info = env.reset()
print(f"Initial actions: {env.game.state.playable_actions[:2]}")
mask = env.get_valid_actions_mask()
valid_indices = np.where(mask == 1)[0]

if len(valid_indices) > 0:
    action = valid_indices[0]
    print(f"Taking action: {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"STEP_1_REWARD: {reward}, TERMINATED: {terminated}")
    
    if not terminated:
        playable = env.game.state.playable_actions
        print(f"Step 2 Playable actions (First 5): {playable[:5]}")
        for a in playable[:5]:
            print(f"Action: {a}, Enum: {a[1]}, Val Type: {type(a[2])}")
        
        try:
            mask2 = env.get_valid_actions_mask()
            print("Mask 2 generated successfully")
        except Exception as e:
            print(f"Mask 2 FAILED: {e}")
            import traceback
            traceback.print_exc()
else:
    print("NO_INITIAL_ACTIONS")
