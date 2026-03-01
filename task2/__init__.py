"""
Task 2: Reinforcement Learning - Opponent Modelling Variant

Complete implementation for training, evaluating, and analyzing PPO agents
in the Chef's Hat Gym environment with focus on opponent modelling.

Student ID: 16946378 (ID mod 7 = 1)
Variant: Opponent Modelling (ID mod 7 = 0 or 1)

Main modules:
- train_ppo: PPO training with opponent tracking
- evaluate: Comprehensive evaluation framework
- opponent_modeller: Opponent behavior analysis
- environment_wrapper: Enhanced Chef's Hat environment
- experiments: Experiment orchestration
- config: Configuration management
- utils: Utility functions and plotting
"""

__version__ = "1.0.0"
__author__ = "Task 2 Implementation"

from .train_ppo import PPOTrainer
from .evaluate import EvaluationEngine
from .opponent_modeller import OpponentModeller, OpponentProfile
from .environment_wrapper import ChefsHatWrapper, GymCompatibleChefsHat
from .config import get_config, get_all_experiments

__all__ = [
    'PPOTrainer',
    'EvaluationEngine',
    'OpponentModeller',
    'OpponentProfile',
    'ChefsHatWrapper',
    'GymCompatibleChefsHat',
    'get_config',
    'get_all_experiments',
]
