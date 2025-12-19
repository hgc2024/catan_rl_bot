from collections import Counter

class ResourceTracker:
    def __init__(self, num_players=4):
        self.num_players = num_players
        # Track 5 resources: WOOD, BRICK, SHEEP, WHEAT, ORE
        # We'll map them to indices 0-4
        self.resource_map = {
            "WOOD": 0, "BRICK": 1, "SHEEP": 2, "WHEAT": 3, "ORE": 4
        }
        
    def reset(self):
        pass

    def update_from_game_state(self, game_state):
        pass
        
    def get_opponent_resources(self, game_state, current_player_id):
        """
        Returns a flat list of opponent resource counts using perfect information 
        from game_state (cheating as per Phase 1 simplification).
        
        Args:
            game_state: catanatron.models.State object
            current_player_id: int (0-3)
            
        Returns:
            np.array or list of shape (15,) -> 3 opponents * 5 resources
        """
        opponents_vectors = []
        
        # Relative indices: (id+1)%4, (id+2)%4, (id+3)%4
        opponent_ids = [(current_player_id + i) % self.num_players for i in range(1, 4)]
        
        # Access player_state dict from game_state
        # Catanatron stores resources as P{id}_{RES}_IN_HAND
        player_state = getattr(game_state, "player_state", {})
        
        for pid in opponent_ids:
            p_res_vec = []
            for res_name in ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]:
                key = f"P{pid}_{res_name}_IN_HAND"
                count = player_state.get(key, 0)
                p_res_vec.append(count)
            
            opponents_vectors.extend(p_res_vec)
            
        return opponents_vectors
