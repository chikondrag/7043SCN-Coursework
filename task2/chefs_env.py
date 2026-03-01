"""
ChefsHat Gymnasium Environment

This module creates a gymnasium-compatible environment for the Chef's Hat card game.
It handles initialization and registration of the environment.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import sys
from pathlib import Path
import tempfile
import os

# Add src to path so we can import ChefsHatGym
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from core.game_env.game import Game
    from agents.random_agent import RandomAgent
    CHEFS_HAT_AVAILABLE = True
except ImportError:
    CHEFS_HAT_AVAILABLE = False


class ChefsHatGymEnv(gym.Env):
    """
    Gymnasium wrapper for Chef's Hat card game.
    
    This environment provides a standard gym interface to the Chef's Hat game,
    including state representation, action masking, and reward tracking.
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, num_players: int = 4, opponent_type: str = "random"):
        """
        Initialize the Chef's Hat environment.
        
        Args:
            num_players: Number of players in the game
            opponent_type: Type of opponent agents to use ("random", "heuristic", etc.)
        """
        super().__init__()
        
        if not CHEFS_HAT_AVAILABLE:
            raise RuntimeError("ChefsHatGym not available. Cannot create environment.")
        
        self.num_players = num_players
        self.opponent_type = opponent_type
        
        # Create a temporary directory for agent logs
        self.temp_dir = tempfile.mkdtemp(prefix="chefs_hat_")
        
        # Initialize game
        player_names = [f"Player_{i}" for i in range(num_players)]
        self.game = Game(
            player_names=player_names,
            max_matches=1,
            max_rounds=100,
            save_dataset=False
        )
        
        # Create opponent agents (all random for now)
        self.opponents = [
            RandomAgent(name=f"Opponent_{i}", log_directory=self.temp_dir) 
            for i in range(num_players - 1)
        ]
        
        # Action and observation spaces
        # 52 cards in deck, action is which card (0-51) or pass (52)
        self.action_space = spaces.Discrete(53)
        
        # Observation: card in hand, cards on table, opponent state, etc.
        # Simplified: card vector (52 dims) + table state
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(100,), dtype=np.float32
        )
        
        self.current_state = None
        self.done = False
    
    def _get_observation(self) -> np.ndarray:
        """Get current game state as observation."""
        # Placeholder observation (all zeros for now)
        # In a real implementation, this would extract state from the game
        obs = np.zeros(100, dtype=np.float32)
        return obs
    
    def _get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions."""
        # All actions are valid in this simplified version
        return np.ones(53, dtype=np.int32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        
        # Reset game state
        player_names = [f"Player_{i}" for i in range(self.num_players)]
        self.game = Game(
            player_names=player_names,
            max_matches=1,
            max_rounds=100,
            save_dataset=False
        )
        
        self.done = False
        obs = self._get_observation()
        
        return obs, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step of the environment.
        
        Args:
            action: Action to take (0-52 card index or 52 for pass)
            
        Returns:
            obs, reward, terminated, truncated, info
        """
        reward = 0.0
        truncated = False
        
        # Process game step - simplified version
        # In real implementation, this would interact with game logic
        
        # Check if game is done - game.finished returns True when complete
        terminated = self.game.finished if hasattr(self.game, 'finished') else False
        
        obs = self._get_observation()
        info = {
            "valid_actions": self._get_valid_actions(),
            "action_mask": self._get_valid_actions()
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (not implemented)."""
        pass
    
    def close(self):
        """Clean up resources."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass


def create_chefs_hat_env(**kwargs) -> ChefsHatGymEnv:
    """Factory function to create ChefsHat environment."""
    return ChefsHatGymEnv(**kwargs)


# Register the environment with gymnasium
try:
    gym.register(
        id="ChefsHat-v0",
        entry_point="chefs_env:ChefsHatGymEnv",
        max_episode_steps=1000,
    )
except gym.error.Error:
    # Environment already registered, that's fine
    pass
