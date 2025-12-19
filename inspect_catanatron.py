try:
    from catanatron import Game, Color
    from catanatron.models.map import CatanMap
    from catanatron.models.player import Player
    
    print("--- Catanatron Imported Successfully ---")
    game = Game()
    state = game.state
    
    print(f"State Attributes: {dir(state)}")
    print(f"Board (Map): {state.board}")
    print(f"Map Attributes: {dir(state.board)}")
    
    # Check how to access hexes
    # usually state.board.hexes or similar
    # Check how to access nodes/edges
    
    print(f"Players: {state.players}")
    # Check player attributes
    p0 = state.players[0]
    print(f"Player Attributes: {dir(p0)}")
    print(f"Resources: {p0.resources}")
    
except ImportError:
    print("Catanatron not installed yet.")
except Exception as e:
    print(f"Error: {e}")
