from catanatron import Game, Color
from catanatron.models.player import Player

players = [Player(Color.RED), Player(Color.BLUE), Player(Color.WHITE), Player(Color.ORANGE)]
game = Game(players)
state = game.state

print("State attributes:")
for d in dir(state):
    print(d)
    
print("-" * 20)
print("Player 0 attributes:")
p0 = state.players[0]
for d in dir(p0):
    print(d)

print("-" * 20)
print("State attributes values:")
# print(state.__dict__) # Might be too big
