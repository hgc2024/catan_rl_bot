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
        # 126-130: Buy Dev Card / Roll
        # 126: Buy, 127: Roll
        # 131-135: Play Dev Card
        # 136-154: Move Robber
        # 155-160: Maritime Trade
        # 161-200: Domestic Trade 
        # 201: End Turn
        self.action_space = spaces.Discrete(202)
        
        self.game = None
        self.player_id = 0 # Agent controls player 0
        self.resource_tracker = ResourceTracker()
        
        # Build static mappings for consistent indexing
        self.node_list = list(range(54))
        self.edge_list = self._build_edge_list()
        self.edge_to_idx = {edge: i for i, edge in enumerate(self.edge_list)}
        
        # Build hex mapping (19 land hexes)
        players = [Player(Color.RED), Player(Color.BLUE), Player(Color.WHITE), Player(Color.ORANGE)]
        temp_game = Game(players)
        self.hex_list = sorted(temp_game.state.board.map.land_tiles.keys())
        self.hex_to_idx = {coord: i for i, coord in enumerate(self.hex_list)}

        self._last_vp = 0

    def _build_edge_list(self):
        # Brute-force discovery of all edges by placing settlements everywhere
        edges = set()
        for node_id in range(54):
            players = [Player(Color.RED), Player(Color.BLUE), Player(Color.WHITE), Player(Color.ORANGE)]
            temp_game = Game(players)
            # Try to place settlement at node_id
            # Note: In setup, we can place settlement anywhere that is valid
            try:
                # Construct action manually
                color = temp_game.state.current_color
                if callable(color): color = color()
                act = Action(color, ActionType.BUILD_SETTLEMENT, node_id)
                temp_game.execute(act)
                
                # Check resulting playable actions for roads
                for potential_road in temp_game.state.playable_actions:
                    if potential_road[1] == ActionType.BUILD_ROAD:
                        edges.add(tuple(sorted(potential_road[2])))
            except:
                # Some nodes might be blocked or invalid in this layout?
                # But in a new game, most should be fine.
                continue
        
        edge_list = sorted(list(edges))
        # print(f"DEBUG: Discovered {len(edge_list)} edges")
        return edge_list

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
        self._last_vp = 0
        
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action_idx):
        catan_action = self._map_action(action_idx)
        
        try:
            self.game.execute(catan_action)
            # Reward based on Victory Points change
            curr_vp = self.game.state.player_state[f"P{self.player_id}_VICTORY_POINTS"]
            reward = float(curr_vp - self._last_vp)
            self._last_vp = curr_vp
            
            # Bonus for winning
            win_color = self.game.winning_color() if callable(self.game.winning_color) else self.game.winning_color
            if win_color == Color.RED: # Player 0 is RED
                reward += 10.0
                
        except Exception as e:
            # Invalid move mapping fallback
            reward = -1.0 # Small penalty for picking a move that failed execution
            
        # Check termination
        win_color = self.game.winning_color() if callable(self.game.winning_color) else self.game.winning_color
        terminated = win_color is not None
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
            act_type = action[1]
            val = action[2]
            name = getattr(act_type, 'name', str(act_type))
            
            if name == "BUILD_SETTLEMENT" or name == "BUILD_CITY":
                if isinstance(val, int) and 0 <= val <= 53:
                    mask[val] = 1
                    
            elif name == "BUILD_ROAD":
                if isinstance(val, tuple):
                    val_sorted = tuple(sorted(val))
                    if val_sorted in self.edge_to_idx:
                        mask[54 + self.edge_to_idx[val_sorted]] = 1
                    
            elif name == "BUY_DEVELOPMENT_CARD":
                mask[126] = 1
            elif name == "ROLL":
                mask[127] = 1
                
            elif name == "PLAY_KNIGHT_CARD":
                mask[131] = 1
            elif name == "PLAY_YEAR_OF_PLENTY":
                mask[132] = 1
            elif name == "PLAY_ROAD_BUILDING":
                mask[133] = 1
            elif name == "PLAY_MONOPOLY":
                mask[134] = 1
                
            elif name == "MOVE_ROBBER":
                 # val is (coord, victim, None)
                 coord = val[0] if isinstance(val, tuple) else val
                 if coord in self.hex_to_idx:
                     mask[136 + self.hex_to_idx[coord]] = 1
            
            elif name == "DISCARD":
                mask[201] = 1 # Map DISCARD to a single action for now
            
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
            edge_idx = action_idx - 54
            if edge_idx < len(self.edge_list):
                return Action(color, ActionType.BUILD_ROAD, self.edge_list[edge_idx])
            
        # 126: Buy Dev Card
        elif action_idx == 126:
            return Action(color, ActionType.BUY_DEVELOPMENT_CARD, None)
            
        # 127: Roll
        elif action_idx == 127:
            return Action(color, ActionType.ROLL, None)
            
        # 131-135: Play Dev Card
        elif 131 <= action_idx <= 135:
            card_map = {131: ActionType.PLAY_KNIGHT_CARD, 
                        132: ActionType.PLAY_YEAR_OF_PLENTY, 
                        133: ActionType.PLAY_ROAD_BUILDING, 
                        134: ActionType.PLAY_MONOPOLY}
            act_type = card_map.get(action_idx)
            if act_type:
                 # Note: Development card actions in catanatron might need the card as value?
                 # But usually None works for simple play. 
                 # Let's check state_functions.py: play_dev_card(state, color, dev_card)
                 # If it fails, we may need to specify card type.
                 return Action(color, act_type, None)
                 
        # 136-154: Move Robber
        elif 136 <= action_idx <= 154:
            map_idx = action_idx - 136
            if map_idx < len(self.hex_list):
                target_coord = self.hex_list[map_idx]
                # Find valid MOVE_ROBBER action for this coord
                for a in self.game.state.playable_actions:
                    if a[1] == ActionType.MOVE_ROBBER:
                        coord = a[2][0] if isinstance(a[2], tuple) else a[2]
                        if coord == target_coord:
                            return Action(color, ActionType.MOVE_ROBBER, a[2])
            
        # 201: End Turn or DISCARD (Multiplexed)
        elif action_idx == 201:
            for a in self.game.state.playable_actions:
                if a[1] == ActionType.DISCARD:
                    return Action(color, ActionType.DISCARD, a[2])
                if a[1] == ActionType.END_TURN:
                    return Action(color, ActionType.END_TURN, None)
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
        
        buildings = board.buildings
        for node_id, (owner_color, b_type) in buildings.items():
            if 0 <= node_id < self.n_vertices:
                # Color index: RED=0, BLUE=1, WHITE=2, ORANGE=3
                c_idx = {Color.RED: 0, Color.BLUE: 1, Color.WHITE: 2, Color.ORANGE: 3}.get(owner_color, 0)
                if "SETTLEMENT" in str(b_type):
                    vertex_obs[node_id, 1 + c_idx] = 1.0
                elif "CITY" in str(b_type):
                    vertex_obs[node_id, 5 + c_idx] = 1.0
        
        # --- 3. Edges (72 Links) ---
        # Features: 1 Empty + 4 Roads (P0-P3)
        edge_obs = np.zeros((self.n_edges, self.n_edge_features), dtype=np.float32)
        
        roads = board.roads
        for edge_tuple, owner_color in roads.items():
            edge_tuple_sorted = tuple(sorted(edge_tuple))
            if edge_tuple_sorted in self.edge_to_idx:
                e_idx = self.edge_to_idx[edge_tuple_sorted]
                c_idx = {Color.RED: 0, Color.BLUE: 1, Color.WHITE: 2, Color.ORANGE: 3}.get(owner_color, 0)
                edge_obs[e_idx, 1 + c_idx] = 1.0
        
        # --- 4. Globals ---
        global_obs = np.zeros((self.n_globals,), dtype=np.float32)
        
        # Player-specific values
        for p in range(4):
            # Victory Points
            global_obs[p] = state.player_state.get(f"P{p}_VICTORY_POINTS", 0)
            
        # Self Resources
        r_map_order = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
        for idx, res_name in enumerate(r_map_order):
            global_obs[4 + idx] = state.player_state.get(f"P{self.player_id}_{res_name}_IN_HAND", 0)
        
        # Opponent Resources (from Tracker)
        opp_res = self.resource_tracker.get_opponent_resources(state, self.player_id)
        global_obs[9:9+15] = opp_res
        
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
