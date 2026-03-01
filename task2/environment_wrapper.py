"""
Chef's Hat Gym Environment Wrapper for RL Training

This wrapper provides:
1. Proper state representation for RL agents
2. Action masking (only valid actions)
3. Reward shaping options
4. Opponent information tracking
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Import chefs_env to ensure environment is registered
try:
    import chefs_env
except ImportError:
    pass


class ChefsHatWrapper(gym.Wrapper):
    """
    Wrapper for Chef's Hat Gym environment to facilitate RL training.
    
    Key Features:
    - State representation including own hand, table state, opponent info
    - Action masking for invalid actions
    - Reward tracking and shaping
    - Opponent behavior tracking (for opponent modelling)
    """
    
    def __init__(self, env, opponent_modelling: bool = True, reward_shaping: str = "none"):
        """
        Args:
            env: Base Chef's Hat Gym environment
            opponent_modelling: Whether to track opponent information
            reward_shaping: Type of reward shaping ("none", "win_bonus", "action_penalty")
        """
        super().__init__(env)
        self.opponent_modelling = opponent_modelling
        self.reward_shaping = reward_shaping
        
        # Track game state
        self.episode_rewards = 0.0
        self.episode_step_count = 0
        self.opponent_actions = [[] for _ in range(3)]  # Track last N actions per opponent
        self.opponent_history = {
            'win_count': [0, 0, 0],
            'action_counts': [{} for _ in range(3)],
        }
        
    def reset(self, seed=None, options=None):
        """Reset environment and tracking variables."""
        obs, info = self.env.reset(seed=seed, options=options)
        self.episode_rewards = 0.0
        self.episode_step_count = 0
        self.opponent_actions = [[] for _ in range(3)]
        return obs, info
    
    def step(self, action: int):
        """
        Execute action and return observation with optional reward shaping.
        
        Returns:
            obs: Enhanced observation
            reward: Shaped reward
            terminated: Episode finished
            truncated: Time limit reached
            info: Additional information
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_step_count += 1
        base_reward = reward
        shaped_reward = self._shape_reward(base_reward, terminated, action, info)
        self.episode_rewards += shaped_reward
        
        # Track opponent actions if available
        if self.opponent_modelling and 'opponent_actions' in info:
            self._track_opponent_actions(info)
        
        return obs, shaped_reward, terminated, truncated, info
    
    def _shape_reward(self, base_reward: float, terminated: bool, 
                      action: int, info: Dict) -> float:
        """
        Apply reward shaping based on configuration.
        
        Shaping options:
        - "none": No shaping, use raw rewards
        - "win_bonus": Add bonus for winning
        - "action_penalty": Penalize invalid actions
        """
        if self.reward_shaping == "none":
            return base_reward
        
        shaped = base_reward
        
        if self.reward_shaping in ["win_bonus", "all"]:
            if terminated and base_reward > 0:
                shaped += 5.0  # Bonus for winning
        
        if self.reward_shaping in ["action_penalty", "all"]:
            # Penalty for potentially poor actions (0 reward when no progress)
            if base_reward == 0 and self.episode_step_count > 100:
                shaped -= 0.01
        
        return shaped
    
    def _track_opponent_actions(self, info: Dict):
        """Track opponent actions for opponent modelling."""
        try:
            if 'opponent_actions' in info:
                opponent_acts = info['opponent_actions']
                for opp_idx, action in enumerate(opponent_acts):
                    if opp_idx < 3:  # 3 other players
                        self.opponent_actions[opp_idx].append(action)
                        # Keep only last 10 actions
                        if len(self.opponent_actions[opp_idx]) > 10:
                            self.opponent_actions[opp_idx].pop(0)
        except Exception as e:
            pass  # Silently continue if opponent tracking fails
    
    def get_opponent_features(self) -> np.ndarray:
        """
        Get opponent modelling features from tracked history.
        
        Features:
        - Average action length (exploration)
        - Win rate
        - Most common action type
        """
        features = []
        for opp_idx in range(3):
            actions = self.opponent_actions[opp_idx]
            win_rate = (self.opponent_history['win_count'][opp_idx] / 
                       max(1, sum(self.opponent_history['win_count'])))
            
            # Action length stats
            avg_action_len = np.mean([len(a) if isinstance(a, (list, tuple)) else 1 
                                     for a in actions]) if actions else 0
            
            features.extend([win_rate, avg_action_len, len(actions)])
        
        return np.array(features, dtype=np.float32)
    
    def get_game_stats(self) -> Dict[str, float]:
        """Get current game statistics."""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_step_count,
            'opponent_actions_tracked': sum(len(a) for a in self.opponent_actions),
        }


class GymCompatibleChefsHat(gym.Env):
    """
    Create a fully Gym-compatible interface for Chef's Hat.
    This creates a simplified version for easier RL training.
    """
    
    def __init__(self, num_agents: int = 4, use_opponent_modelling: bool = True):
        """
        Initialize the environment.
        
        Args:
            num_agents: Number of agents (typically 4 for Chef's Hat)
            use_opponent_modelling: Include opponent state in observations
        """
        super().__init__()
        
        try:
            import ChefsHatGym
            self.base_env = gym.make("ChefsHat-v0")
        except Exception as e:
            raise RuntimeError(f"Failed to create ChefsHatGym environment: {e}")
        
        self.num_agents = num_agents
        self.use_opponent_modelling = use_opponent_modelling
        self.agent_idx = 0  # Trained agent is player 0
        
        # Define action and observation spaces
        self.action_space = self.base_env.action_space
        
        # Observation space: flattened state (hand + table + optional opponent features)
        obs_size = self._estimate_obs_size()
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        self.last_obs = None
        self.valid_actions_mask = None
        self.opponent_modeller = None
        
    def _estimate_obs_size(self) -> int:
        """Estimate observation space size."""
        # Base observation size (depends on actual environment)
        base_size = 200  # Adjust based on actual obs from ChefsHatGym
        opponent_features = 9 if self.use_opponent_modelling else 0  # 3 opponents × 3 features
        return base_size + opponent_features
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        obs, info = self.base_env.reset()
        self.last_obs = self._process_observation(obs, info)
        
        if seed is not None:
            np.random.seed(seed)
        
        return self.last_obs, info
    
    def step(self, action: int):
        """Execute one step."""
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        processed_obs = self._process_observation(obs, info)
        
        # Extract valid actions mask if available
        if 'valid_actions' in info:
            self.valid_actions_mask = info['valid_actions']
        
        return processed_obs, reward, terminated, truncated, info
    
    def _process_observation(self, obs, info: Dict) -> np.ndarray:
        """Convert raw observation to standardized format."""
        if isinstance(obs, np.ndarray):
            processed = obs.flatten().astype(np.float32)
        else:
            # Convert to numpy if needed
            processed = np.array(obs, dtype=np.float32).flatten()
        
        # Add opponent modelling features if enabled
        if self.use_opponent_modelling and 'opponent_info' in info:
            opponent_features = self._extract_opponent_features(info)
            processed = np.concatenate([processed, opponent_features])
        
        # Pad or trim to observation space size
        if len(processed) < self.observation_space.shape[0]:
            processed = np.pad(processed, 
                              (0, self.observation_space.shape[0] - len(processed)),
                              mode='constant')
        else:
            processed = processed[:self.observation_space.shape[0]]
        
        return processed
    
    def _extract_opponent_features(self, info: Dict) -> np.ndarray:
        """Extract opponent modelling features from info."""
        features = []
        
        try:
            opponent_info = info.get('opponent_info', {})
            for i in range(3):
                # Win rate, average hand size, action diversity
                win_rate = opponent_info.get(f'opponent_{i}_wins', 0) / max(1, sum(
                    opponent_info.get(f'opponent_{j}_wins', 0) for j in range(3)
                ))
                hand_size = opponent_info.get(f'opponent_{i}_hand_size', 0) / 14.0  # Max hand size
                action_entropy = opponent_info.get(f'opponent_{i}_action_entropy', 0) / 5.0  # Normalize
                
                features.extend([win_rate, hand_size, action_entropy])
        except Exception:
            features = [0.0] * 9  # Default: no opponent info
        
        return np.array(features, dtype=np.float32)
    
    def render(self):
        """Render the environment."""
        self.base_env.render()
    
    def close(self):
        """Close the environment."""
        self.base_env.close()


class ActionMaskWrapper(gym.Wrapper):
    """Wrapper to handle action masking for invalid actions."""
    
    def step(self, action):
        """Step with action validation."""
        # Ensure action is valid
        if not self._is_valid_action(action):
            # Choose random valid action
            valid_actions = self._get_valid_actions()
            if valid_actions:
                action = np.random.choice(valid_actions)
        
        return self.env.step(action)
    
    def _is_valid_action(self, action: int) -> bool:
        """Check if action is valid."""
        try:
            if hasattr(self.env, 'valid_actions_mask'):
                if isinstance(self.env.valid_actions_mask, np.ndarray):
                    return bool(self.env.valid_actions_mask[action])
        except Exception:
            return True
        return True
    
    def _get_valid_actions(self) -> List[int]:
        """Get list of valid actions."""
        try:
            if hasattr(self.env, 'valid_actions_mask'):
                mask = self.env.valid_actions_mask
                if isinstance(mask, np.ndarray):
                    return list(np.where(mask)[0])
        except Exception:
            pass
        return list(range(self.action_space.n))
