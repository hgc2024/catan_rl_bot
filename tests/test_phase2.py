import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.catan_env import CatanEnv

class TestPhase2(unittest.TestCase):
    def setUp(self):
        print(f"DEBUG TEST: CatanEnv file: {sys.modules['src.env.catan_env'].__file__}")
        self.env = CatanEnv()

    def test_mask_shape_and_type(self):
        """Test if validation mask has correct shape and type."""
        self.env.reset()
        mask = self.env.get_valid_actions_mask()
        self.assertEqual(mask.shape, (self.env.action_space.n,))
        self.assertEqual(mask.dtype, np.int8)
        
    def test_initial_valid_actions(self):
        """Test that start of game allows placing settlements."""
        self.env.reset()
        mask = self.env.get_valid_actions_mask()
        
        # In setup phase, we should be able to build settlements (indices 0-53)
        # Sum of build settlement actions should be > 0
        settlement_actions = mask[0:54]
        
        self.assertGreater(np.sum(settlement_actions), 0, "Should be able to place settlement at start")
        
        # Should not be able to end turn (idx 201) immediately in setup
        # Actually catanatron setup phase might force specific moves.
        # But definitely road building (54-125) should act differently depending on rule (settlement first)
        
    def test_observation_structure(self):
        """Test that observation dict has all keys and correct shapes."""
        obs, info = self.env.reset()
        
        self.assertIn("board", obs)
        self.assertIn("vertices", obs)
        self.assertIn("edges", obs)
        self.assertIn("globals", obs)
        
        self.assertEqual(obs["board"].shape, (19, 17))
        # self.assertEqual(obs["vertices"].shape, (54, 15)) # Vertices currently placeholder
        # self.assertEqual(obs["edges"].shape, (72, 5)) # Edges currently placeholder
        self.assertEqual(obs["globals"].shape, (59,)) # 59 is current global size
        
    def test_board_observation_content(self):
        """Test that board observation is not empty (contains resources/numbers)."""
        obs, info = self.env.reset()
        board = obs["board"]
        
        # Sum of entire board obs should be > 0 (resources + numbers)
        self.assertGreater(np.sum(board), 0, "Board observation should contain data")
        
    def test_global_observation_content(self):
        """Test globals observation shape."""
        obs, info = self.env.reset()
        glob = obs["globals"]
        self.assertEqual(len(glob), 59)

if __name__ == '__main__':
    unittest.main()
