from catanatron import Game, Player, Color
game = Game([Player(Color.RED), Player(Color.BLUE)])
playable = game.state.playable_actions

for a in playable:
    if "BUILD_SETTLEMENT" in str(a[1]):
        node = a[2]
        print(f"Node: {node}, Type: {type(node)}")
        print(f"Node dir: {dir(node)}")
        # Check for id, index, or value
        for attr in ['id', 'index', 'value', 'node_id']:
            if hasattr(node, attr):
                print(f"  {attr}: {getattr(node, attr)}")
        break

# Place settlement to get roads
game.execute(playable[0])
playable2 = game.state.playable_actions
for a in playable2:
    if "BUILD_ROAD" in str(a[1]):
        edge = a[2]
        print(f"Edge: {edge}, Type: {type(edge)}")
        print(f"Edge dir: {dir(edge)}")
        break
