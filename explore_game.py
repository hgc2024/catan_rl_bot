from catanatron import Game, Color
from catanatron.models.player import Player

players = [Player(Color.RED), Player(Color.BLUE), Player(Color.WHITE), Player(Color.ORANGE)]
game = Game(players)

print("Game attributes:")
for d in dir(game):
    print(d)
