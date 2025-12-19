from catanatron import Game, Color
from catanatron.models.player import Player

players = [Player(Color.RED), Player(Color.BLUE), Player(Color.WHITE), Player(Color.ORANGE)]
game = Game(players)

print("Game created.")
print(f"Playable Actions: {game.state.playable_actions}")
print(f"Current Color: {game.state.current_color}")
print(f"Winning Color: {game.winning_color}")

import numpy as np

mask = np.zeros(202, dtype=np.int8)
playable = game.state.playable_actions

for action in playable:
    try:
        act_type = action[1]
        val = action[2]
    except:
        continue

    name = getattr(act_type, 'name', str(act_type))
    # print(f"Processing: {name}, Val: {val}")
    
    if name == "BUILD_SETTLEMENT" or name == "BUILD_CITY":
        if val is not None and 0 <= val <= 53:
            mask[val] = 1
            
    elif name == "BUILD_ROAD":
        if val is not None and 0 <= val <= 71: 
            mask[54 + val] = 1

print(f"Mask Ones Indices: {np.where(mask == 1)[0]}")
print(f"Mask Sum: {np.sum(mask)}")
