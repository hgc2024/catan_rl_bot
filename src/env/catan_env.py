import gymnasium as gym
import numpy as np
from gymnasium import spaces
from catanatron import Game, Color, Action
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
        self.game = Game()
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
            # Possibly terminate if too many invalid?
            
        # Check termination
        terminated = self.game.state.is_game_over
        truncated = False
        
        self.resource_tracker.update_from_game_state(self.game.state)
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info

    def _map_action(self, action_idx):
        # 0-53: Build Settlement/City (Vertex)
        if 0 <= action_idx <= 53:
            node_id = action_idx
            # Check if settlement exists to decide City vs Settlement
            # This requires reading state. Simpler: Try one, fallback? 
            # Or use explicit check.
            # Assuming Settlement for now.
            return Action(self.game.state.current_color, "BUILD_SETTLEMENT", node_id)
            
        # 54-125: Build Road (Edge)
        elif 54 <= action_idx <= 125:
            edge_id = action_idx - 54
            return Action(self.game.state.current_color, "BUILD_ROAD", edge_id)
            
        # 126: Buy Dev Card
        elif action_idx == 126:
            return Action(self.game.state.current_color, "BUY_DEVELOPMENT_CARD", None)
            
        # 131-135: Play Dev Card
        elif 131 <= action_idx <= 135:
            card_map = {131: "KNIGHT", 132: "YEAR_OF_PLENTY", 133: "ROAD_BUILDING", 134: "MONOPOLY"}
            card_type = card_map.get(action_idx)
            if card_type:
                 return Action(self.game.state.current_color, "PLAY_" + card_type, None)
                 
        # 136-154: Move Robber
        elif 136 <= action_idx <= 154:
            hex_id = action_idx - 136
            return Action(self.game.state.current_color, "MOVE_ROBBER", hex_id)
            
        # 201: End Turn
        elif action_idx == 201:
            return Action(self.game.state.current_color, "END_TURN", None)
            
        # Fallback or Todo: Trade
        return Action(self.game.state.current_color, "END_TURN", None)

    def _get_obs(self):
        # Implement full feature extraction
        # For Phase 1 skeleton, we return zeros but with correct shapes
        return {
            "board": np.zeros((self.n_hexes, self.n_hex_features), dtype=np.float32),
            "vertices": np.zeros((self.n_vertices, self.n_vertex_features), dtype=np.float32),
            "edges": np.zeros((self.n_edges, self.n_edge_features), dtype=np.float32),
            "globals": np.zeros((self.n_globals,), dtype=np.float32),
        }

    def _get_info(self):
        return {}

    def render(self):
        pass
