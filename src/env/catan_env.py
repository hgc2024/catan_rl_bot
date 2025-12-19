import gymnasium as gym
import numpy as np
from gymnasium import spaces
from catanatron import Game, Color, Action
from catanatron.models.enums import ActionType
from catanatron.models.player import Player
from .wrappers.resource_tracker import ResourceTracker

class CatanEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        
        # --- Observation Space ---
        # 1. Board Grid (19 Hexes)
        # Features: 6 Resources + 10 Numbers + 1 Robber = 17
        self.n_hexes = 19
        self.n_hex_features = 6 + 10 + 1 
        
        # 2. Vertices (54 Nodes)
        # Features: 1 Empty + 4 Settlements + 4 Cities + 6 Port Types = 15
        self.n_vertices = 54
        self.n_vertex_features = 1 + 4 + 4 + 6
        
        # 3. Edges (72 Links)
        # Features: 1 Empty + 4 Roads = 5
        self.n_edges = 72
        self.n_edge_features = 1 + 4
        
        # 4. Globals
        # VPs (4), Public Dev Played (5 types * 4 players = 20), 
        # Longest Road (5), Largest Army (5), 
        # Resources (Self: 5, Opponents: 15) -> 20
        # Dev Cards (Self Unplayed: 5)
        # Total approx: 4 + 20 + 5 + 5 + 20 + 5 = 59
        self.n_globals = 59 

        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(self.n_hexes, self.n_hex_features), dtype=np.float32),
            "vertices": spaces.Box(low=0, high=1, shape=(self.n_vertices, self.n_vertex_features), dtype=np.float32),
            "edges": spaces.Box(low=0, high=1, shape=(self.n_edges, self.n_edge_features), dtype=np.float32),
            "globals": spaces.Box(low=0, high=float('inf'), shape=(self.n_globals,), dtype=np.float32),
        })

        # --- Action Space ---
        # Flattened Discrete Action Space
        # 0-53: Build Settlement/City (Vertex)
        # 54-125: Build Road (Edge)
        # 126-130: Buy Dev Card
        # 131-135: Play Dev Card
        # 136-154: Move Robber
        # 155-160: Maritime Trade
        # 161-200: Domestic Trade 
        # 201: End Turn
        self.action_space = spaces.Discrete(202)
        
        self.game = None
        self.player_id = 0 # Agent controls player 0
        self.resource_tracker = ResourceTracker()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        players = [
            Player(Color.RED),
            Player(Color.BLUE),
            Player(Color.WHITE),
            Player(Color.ORANGE),
        ]
        self.game = Game(players)
        self.resource_tracker.reset()
        
        # Fast forward if we need to (or valid start)
        # Catanatron starts at setup phase. 
        # For RL, we might want to play setup automatically or let agent do it.
        # Assuming agent does setup too (since actions include building).
        
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action_idx):
        catan_action = self._map_action(action_idx)
        
        # Apply action
        # Catanatron's game.play() applies action and advances turn logic
        # current_color = self.game.state.current_color
        # But we need to ensure it's our turn or handle multi-agent.
        # For Phase 1, we control Player 0. If it's not our turn, we must either:
        # a) Simulate opponents (Random/Heuristic) until it is our turn again.
        # b) Assume self-play environment where we control all? 
        # The prompt says: "Inferred (Opponents)..." implying opponents exist.
        # "Implementation Plan: Phase 2... Masking & Random Agent".
        # We likely need the step() to run until the agent works again.
        
        # NOTE: For now, we apply single action. If returns invalid, we punish.
        # If valid, we verify if turn defines.
        
        # Since catanatron might raise error on invalid, we wrap.
        try:
            self.game.play(catan_action)
            reward = 0 # Calculate based on events (Need Hooks or State Diff)
            # TODO: Implement Reward Shaping based on state diff
        except Exception as e:
            # Invalid move
            reward = -10
            # print(f"DEBUG: Invalid move {action_idx} -> {catan_action}: {e}")
            # terminated = True # Should we terminate on illegal move? SB3 often prefers it.
            
        # Check termination
        terminated = self.game.winning_color is not None
        truncated = False
        
        self.resource_tracker.update_from_game_state(self.game.state)
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info

    def get_valid_actions_mask(self):
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        
        if self.game is None:
            return mask
            
        playable = self.game.state.playable_actions
        
        for action in playable:
            # Action is likely (Color, ActionType, Value)
            # We use index access for safety if field names differ
            # ActionType is an Enum usually.
            
            # Try appropriate access
            try:
                # Assuming namedtuple structure or tuple
                # Index 1 is ActionType
                act_type = action[1]
                val = action[2]
            except:
                continue

            name = getattr(act_type, 'name', str(act_type))
            
            if name == "BUILD_SETTLEMENT" or name == "BUILD_CITY":
                if val is not None and 0 <= val <= 53:
                    mask[val] = 1
                    
            elif name == "BUILD_ROAD":
                if val is not None and 0 <= val <= 71: # 72 edges max?
                    mask[54 + val] = 1
                    
            elif name == "BUY_DEVELOPMENT_CARD":
                mask[126] = 1
                
            elif name == "PLAY_KNIGHT":
                mask[131] = 1
            elif name == "PLAY_YEAR_OF_PLENTY":
                mask[132] = 1
            elif name == "PLAY_ROAD_BUILDING":
                mask[133] = 1
            elif name == "PLAY_MONOPOLY":
                mask[134] = 1
                
            elif name == "MOVE_ROBBER":
                 if val is not None and 0 <= val <= 18:
                     mask[136 + val] = 1
            
            elif name == "END_TURN":
                mask[201] = 1
                
        return mask

    def _map_action(self, action_idx):
        color = self.game.state.current_color
        if callable(color):
            color = color()

        # 0-53: Build Settlement/City (Vertex)
        if 0 <= action_idx <= 53:
            node_id = action_idx
            return Action(color, ActionType.BUILD_SETTLEMENT, node_id)
            
        # 54-125: Build Road (Edge)
        elif 54 <= action_idx <= 125:
            edge_id = action_idx - 54
            return Action(color, ActionType.BUILD_ROAD, edge_id)
            
        # 126: Buy Dev Card
        elif action_idx == 126:
            return Action(color, ActionType.BUY_DEVELOPMENT_CARD, None)
            
        # 131-135: Play Dev Card
        elif 131 <= action_idx <= 135:
            card_map = {131: ActionType.PLAY_KNIGHT, 
                        132: ActionType.PLAY_YEAR_OF_PLENTY, 
                        133: ActionType.PLAY_ROAD_BUILDING, 
                        134: ActionType.PLAY_MONOPOLY}
            act_type = card_map.get(action_idx)
            if act_type:
                 return Action(color, act_type, None)
                 
        # 136-154: Move Robber
        elif 136 <= action_idx <= 154:
            hex_id = action_idx - 136
            return Action(color, ActionType.MOVE_ROBBER, hex_id)
            
        # 201: End Turn
        elif action_idx == 201:
            return Action(color, ActionType.END_TURN, None)
            
        # Fallback
        return Action(color, ActionType.END_TURN, None)

    def _get_obs(self):
        state = self.game.state
        board = state.board
        
        # --- 1. Board Grid (19 Hexes) ---
        # Features: 6 Resources + 10 Numbers + 1 Robber = 17
        board_obs = np.zeros((self.n_hexes, self.n_hex_features), dtype=np.float32)
        
        # Use board.map.tiles if available
        if hasattr(board, 'map') and hasattr(board.map, 'tiles'):
            # Convert values to list and trust order (consistent sort by key)
            hex_items = sorted(board.map.tiles.items()) # List of (coord, hex)
            for i, (coord, hex_obj) in enumerate(hex_items):
                if i >= self.n_hexes: break
                
                # Resource One-Hot (None, Wood, Brick, Sheep, Wheat, Ore) -> 6
                # Hex values: None, WOOD, BRICK...
                res_map = {None: 0, "WOOD": 1, "BRICK": 2, "SHEEP": 3, "WHEAT": 4, "ORE": 5, "DESERT": 0} 
                res_val = getattr(hex_obj, 'resource', None)
                res_idx = res_map.get(res_val, 0)
                board_obs[i, res_idx] = 1.0
                
                # Number Token One-Hot (2-12)
                val = getattr(hex_obj, 'number', None)
                if val and 2 <= val <= 12:
                    board_obs[i, 6 + (val - 2)] = 1.0
                    
                # Robber (Is robber here?)
                # robber_coordinate might be on board?
                if hasattr(board, 'robber_coordinate') and board.robber_coordinate == coord:
                    board_obs[i, 16] = 1.0
                
        # --- 2. Vertices (54 Nodes) ---
        # Features: 1 Empty + 4 Settlements (P0-P3) + 4 Cities (P0-P3) + 6 Port Types = 15
        vertex_obs = np.zeros((self.n_vertices, self.n_vertex_features), dtype=np.float32)
        
        # TODO: Implement robust Node ID -> Coordinate mapping for Catanatron
        # Current Catanatron version doesn't easily expose node list.
        # Leaving as zeros for Phase 2 to prevent crash.
        
        # --- 3. Edges (72 Links) ---
        # Features: 1 Empty + 4 Roads (P0-P3)
        edge_obs = np.zeros((self.n_edges, self.n_edge_features), dtype=np.float32)
        
        # TODO: Implement Edge extraction
        
        # --- 4. Globals ---
        global_obs = np.zeros((self.n_globals,), dtype=np.float32)
        
        # Self Resources
        player_state = getattr(state, "player_state", {})
        r_map_order = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
        for idx, res_name in enumerate(r_map_order):
            key = f"P{self.player_id}_{res_name}_IN_HAND"
            global_obs[idx] = player_state.get(key, 0)
        
        # Opponent Resources (from Tracker)
        opp_res = self.resource_tracker.get_opponent_resources(state, self.player_id)
        offset = 5
        global_obs[offset:offset+15] = opp_res
        
        return {
            "board": board_obs,
            "vertices": vertex_obs,
            "edges": edge_obs,
            "globals": global_obs,
        }


    def _get_info(self):
        return {}

    def render(self):
        pass
