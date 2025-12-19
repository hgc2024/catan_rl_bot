
import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.catan_env import CatanEnv
from src.env.wrappers.resource_tracker import ResourceTracker
from catanatron import Game

class TestPhase1(unittest.TestCase):
    def setUp(self):
        self.env = CatanEnv()

    def test_env_init(self):
        """Test if environment initializes correctly."""
        self.assertIsInstance(self.env, CatanEnv)
        self.assertIsNotNone(self.env.observation_space)
        self.assertIsNotNone(self.env.action_space)

    def test_observation_space(self):
        """Test observation space shapes and types."""
        obs_space = self.env.observation_space
        self.assertTrue(hasattr(obs_space, 'spaces'))
        
        # Check shapes
        self.assertEqual(obs_space['board'].shape, (19, 17))
        self.assertEqual(obs_space['vertices'].shape, (54, 15))
        self.assertEqual(obs_space['edges'].shape, (72, 5))
        self.assertEqual(obs_space['globals'].shape, (59,))

    def test_action_space(self):
        """Test action space size."""
        self.assertEqual(self.env.action_space.n, 202)

    def test_reset(self):
        """Test reset returns correct observation structure."""
        obs, info = self.env.reset()
        
        self.assertIsInstance(obs, dict)
        self.assertIn('board', obs)
        self.assertIn('vertices', obs)
        self.assertIn('edges', obs)
        self.assertIn('globals', obs)
        
        # Check if catanatron game is initialized
        self.assertIsInstance(self.env.game, Game)

    def test_step_structure(self):
        """Test step returns 5-tuple and advances state (smoke test)."""
        self.env.reset()
        action = self.env.action_space.sample()
        
        # Note: This might trigger invalid move punishment (-10) or exception catch
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_resource_tracker(self):
        """Test resource tracker initialization and update."""
        from catanatron import Color
        from catanatron.models.player import Player
        
        players = [Player(Color.RED), Player(Color.BLUE), Player(Color.WHITE), Player(Color.ORANGE)]
        game = Game(players)
        tracker = ResourceTracker()
        tracker.reset()
        
        # Test 0-filled initial state logic (or whatever implementation does)
        # Note: Our current implementation reads directly from game state
        
        opp_resources = tracker.get_opponent_resources(game.state, current_player_id=0)
        # Expect 3 opponents * 5 resources = 15
        self.assertEqual(len(opp_resources), 15)

if __name__ == '__main__':
    unittest.main()
