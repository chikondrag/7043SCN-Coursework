"""
Opponent Modelling Module

This module provides tools for analyzing, tracking, and learning from opponent behavior
in the Chef's Hat game. It's crucial for the Opponent Modelling Variant of Task 2.

Features:
1. Opponent behavior tracking
2. Opponent type classification (Random, Aggressive, Defensive, etc.)
3. Non-stationarity analysis
4. Opponent model visualization
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class OpponentProfile:
    """Profile of an opponent's behavior."""
    
    opponent_id: int
    name: str
    win_rate: float = 0.0
    total_games: int = 0
    total_wins: int = 0
    
    # Action statistics
    action_counts: Dict[str, int] = None
    action_sequences: List[List[int]] = None
    
    # Behavior characteristics
    aggression_score: float = 0.0  # How aggressive in card play
    consistency_score: float = 0.0  # How consistent across games
    average_hand_size: float = 0.0  # Average cards per turn
    
    # Temporal tracking for non-stationarity
    win_rates_over_time: List[float] = None
    behavior_changes: List[float] = None
    
    def __post_init__(self):
        if self.action_counts is None:
            self.action_counts = defaultdict(int)
        if self.action_sequences is None:
            self.action_sequences = []
        if self.win_rates_over_time is None:
            self.win_rates_over_time = []
        if self.behavior_changes is None:
            self.behavior_changes = []
    
    def to_dict(self) -> Dict:
        """Convert profile to dictionary for serialization."""
        return {
            'opponent_id': self.opponent_id,
            'name': self.name,
            'win_rate': float(self.win_rate),
            'total_games': self.total_games,
            'total_wins': self.total_wins,
            'aggression_score': float(self.aggression_score),
            'consistency_score': float(self.consistency_score),
            'average_hand_size': float(self.average_hand_size),
        }


class OpponentModeller:
    """
    Tracks and models opponent behavior over time.
    
    This is the core component for the Opponent Modelling Variant.
    """
    
    def __init__(self, num_opponents: int = 3, window_size: int = 50):
        """
        Initialize opponent modeller.
        
        Args:
            num_opponents: Number of opponents to track (3 for Chef's Hat)
            window_size: Window for computing non-stationarity statistics
        """
        self.num_opponents = num_opponents
        self.window_size = window_size
        
        # Opponent profiles
        self.opponents: Dict[int, OpponentProfile] = {}
        for i in range(num_opponents):
            self.opponents[i] = OpponentProfile(opponent_id=i, name=f"Opponent_{i}")
        
        # Game history
        self.game_history = deque(maxlen=100)
        self.current_game = {
            'actions': defaultdict(list),
            'rewards': defaultdict(float),
            'final_ranks': {},
        }
        
        # Non-stationarity tracking
        self.behavior_windows = [deque(maxlen=window_size) for _ in range(num_opponents)]
    
    def start_game(self):
        """Initialize tracking for a new game."""
        self.current_game = {
            'actions': defaultdict(list),
            'rewards': defaultdict(float),
            'final_ranks': {},
            'action_sequences': defaultdict(list),
        }
    
    def record_action(self, opponent_id: int, action: Any):
        """Record an action taken by an opponent."""
        if opponent_id < self.num_opponents:
            self.current_game['actions'][opponent_id].append(action)
            
            # Track action type if it's numeric
            if isinstance(action, (int, np.integer)):
                action_key = f"action_{action}"
                self.opponents[opponent_id].action_counts[action_key] += 1
    
    def record_reward(self, opponent_id: int, reward: float):
        """Record reward received by opponent."""
        if opponent_id < self.num_opponents:
            self.current_game['rewards'][opponent_id] = reward
    
    def end_game(self, final_ranks: Dict[int, int]):
        """
        Record game completion and update profiles.
        
        Args:
            final_ranks: Dict mapping opponent_id to final rank (1=winner, 4=loser)
        """
        self.current_game['final_ranks'] = final_ranks
        self.game_history.append(dict(self.current_game))
        
        # Update opponent profiles
        for opponent_id in range(self.num_opponents):
            rank = final_ranks.get(opponent_id, 4)
            is_win = (rank == 1)
            
            profile = self.opponents[opponent_id]
            profile.total_games += 1
            if is_win:
                profile.total_wins += 1
            profile.win_rate = profile.total_wins / max(1, profile.total_games)
            
            # Update hand size stats
            num_actions = len(self.current_game['actions'][opponent_id])
            if num_actions > 0:
                profile.average_hand_size = (
                    (profile.average_hand_size * (profile.total_games - 1) + num_actions) /
                    profile.total_games
                )
            
            # Track temporal behavior for non-stationarity
            profile.win_rates_over_time.append(profile.win_rate)
            self.behavior_windows[opponent_id].append({
                'win': is_win,
                'num_actions': num_actions,
                'rank': rank,
            })
    
    def get_opponent_features(self, opponent_id: int) -> np.ndarray:
        """
        Get feature vector representing opponent for model input.
        
        Returns:
            Feature array: [win_rate, avg_hand_size, aggression, consistency, non_stationarity]
        """
        if opponent_id >= self.num_opponents:
            return np.zeros(5, dtype=np.float32)
        
        profile = self.opponents[opponent_id]
        nonstationarity = self._compute_nonstationarity(opponent_id)
        
        features = np.array([
            profile.win_rate,
            profile.average_hand_size / 14.0,  # Normalize by max hand size
            profile.aggression_score,
            profile.consistency_score,
            nonstationarity,
        ], dtype=np.float32)
        
        return features
    
    def get_all_opponent_features(self) -> np.ndarray:
        """Get concatenated features for all opponents."""
        features = []
        for i in range(self.num_opponents):
            features.extend(self.get_opponent_features(i))
        return np.array(features, dtype=np.float32)
    
    def _compute_nonstationarity(self, opponent_id: int) -> float:
        """
        Compute non-stationarity score (variance in behavior over time).
        
        Higher score = more variable behavior = higher non-stationarity
        """
        if len(self.behavior_windows[opponent_id]) < 2:
            return 0.0
        
        recent_behavior = list(self.behavior_windows[opponent_id])
        win_rates = [b['win'] for b in recent_behavior]
        
        # Compute variance in recent win/loss pattern
        if len(win_rates) > 1:
            win_array = np.array(win_rates, dtype=float)
            # Compute moving average variance
            variance = np.var(
                np.convolve(win_array, np.ones(min(5, len(win_array))), mode='valid')
            )
            return float(np.clip(variance, 0, 1))
        
        return 0.0
    
    def get_opponent_type(self, opponent_id: int) -> str:
        """
        Classify opponent type based on behavior.
        
        Types:
        - "random": No clear pattern, high non-stationarity
        - "aggressive": High action count, low win rate
        - "defensive": Low action count, varied win rate
        - "stable": Consistent behavior
        """
        profile = self.opponents[opponent_id]
        nonstationarity = self._compute_nonstationarity(opponent_id)
        
        if nonstationarity > 0.5:
            return "random"
        elif profile.average_hand_size > 5 and profile.win_rate < 0.3:
            return "aggressive"
        elif profile.average_hand_size < 3:
            return "defensive"
        else:
            return "stable"
    
    def get_non_stationarity_report(self) -> Dict[int, Dict[str, Any]]:
        """
        Get detailed non-stationarity analysis for all opponents.
        
        This is key for the Opponent Modelling Variant analysis.
        """
        report = {}
        
        for opponent_id in range(self.num_opponents):
            profile = self.opponents[opponent_id]
            nonstationarity = self._compute_nonstationarity(opponent_id)
            
            # Compute trend (improving or deteriorating)
            win_rates = profile.win_rates_over_time
            if len(win_rates) >= 10:
                early_avg = np.mean(win_rates[:len(win_rates)//2])
                late_avg = np.mean(win_rates[len(win_rates)//2:])
                trend = late_avg - early_avg
            else:
                trend = 0.0
            
            report[opponent_id] = {
                'type': self.get_opponent_type(opponent_id),
                'nonstationarity_score': float(nonstationarity),
                'win_rate': float(profile.win_rate),
                'trend': float(trend),
                'total_games': profile.total_games,
                'consistency_score': float(self._compute_consistency(opponent_id)),
            }
        
        return report
    
    def _compute_consistency(self, opponent_id: int) -> float:
        """
        Compute consistency score (inverse of variability).
        
        Higher = more consistent behavior
        """
        nonstationarity = self._compute_nonstationarity(opponent_id)
        return 1.0 - min(nonstationarity, 1.0)
    
    def get_learning_curves(self) -> Dict[int, List[float]]:
        """Get win rate learning curves for all opponents."""
        curves = {}
        for opponent_id in range(self.num_opponents):
            curves[opponent_id] = self.opponents[opponent_id].win_rates_over_time
        return curves
    
    def export_profiles(self, filepath: str):
        """Export opponent profiles to JSON."""
        profiles = {
            oid: profile.to_dict() 
            for oid, profile in self.opponents.items()
        }
        with open(filepath, 'w') as f:
            json.dump(profiles, f, indent=2)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all opponents."""
        stats = {
            'total_games_recorded': len(self.game_history),
            'opponent_types': {},
            'average_nonstationarity': 0.0,
        }
        
        nonstationarities = []
        for opponent_id in range(self.num_opponents):
            opp_type = self.get_opponent_type(opponent_id)
            stats['opponent_types'][f'opponent_{opponent_id}'] = opp_type
            nonstationarities.append(self._compute_nonstationarity(opponent_id))
        
        stats['average_nonstationarity'] = float(np.mean(nonstationarities))
        
        return stats


class OpponentAdaptationStrategy:
    """
    Strategy for adapting agent behavior based on opponent model.
    """
    
    def __init__(self, modeller: OpponentModeller):
        self.modeller = modeller
        self.strategy_cache = {}
    
    def get_adaptation_bonus(self, opponent_features: np.ndarray) -> float:
        """
        Compute adaptation bonus based on opponent features.
        
        This can be added to rewards to encourage adapting to opponents.
        """
        nonstationarity = opponent_features[4] if len(opponent_features) > 4 else 0
        
        # Bonus for playing against high non-stationary opponents
        adaptation_bonus = nonstationarity * 0.1
        
        return adaptation_bonus
    
    def recommend_strategy(self, opponent_id: int) -> str:
        """
        Recommend strategy against specific opponent.
        
        Returns:
            "aggressive", "defensive", "adaptive", or "passive"
        """
        opp_type = self.modeller.get_opponent_type(opponent_id)
        
        strategies = {
            "random": "adaptive",      # Need to adapt to random behavior
            "aggressive": "defensive", # Defend against aggressive opponents
            "defensive": "aggressive", # Attack defensive opponents
            "stable": "adaptive",      # Adapt to stable patterns
        }
        
        return strategies.get(opp_type, "adaptive")
