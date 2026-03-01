"""
Configuration file for Task 2 experiments

Modify this file to adjust experiment parameters without touching code.
"""

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Training configuration
TRAINING_CONFIG = {
    # Experiment 1: Random opponent
    'exp1_random': {
        'opponent_type': 'random',
        'total_timesteps': 100000,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'save_freq': 5000,
        'eval_freq': 5000,
    },
    
    # Experiment 2: Heuristic opponent
    'exp2_heuristic': {
        'opponent_type': 'heuristic',
        'total_timesteps': 100000,
        'learning_rate': 2e-4,  # Lower for harder opponent
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'save_freq': 5000,
        'eval_freq': 5000,
        'reward_shaping': 'win_bonus',
    },
    
    # Experiment 3: Mixed opponents
    'exp3_mixed': {
        'opponent_type': 'mixed',
        'total_timesteps': 100000,
        'learning_rate': 2.5e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 15,  # More training for complex environment
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'save_freq': 5000,
        'eval_freq': 5000,
        'reward_shaping': 'win_bonus',
    },
}

# Quick testing configuration (for rapid iteration)
QUICK_TEST_CONFIG = {
    'opponent_type': 'random',
    'total_timesteps': 5000,  # Very short for testing
    'learning_rate': 3e-4,
    'n_steps': 512,
    'batch_size': 32,
    'n_epochs': 5,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'save_freq': 2500,
    'eval_freq': 2500,
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVALUATION_CONFIG = {
    'num_episodes': 50,  # Episodes per evaluation
    'cross_eval_opponents': ['random', 'heuristic'],
    'non_stationarity_episodes': 100,
    'eval_seed': 42,
}

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

ENVIRONMENT_CONFIG = {
    'num_agents': 4,
    'use_opponent_modelling': True,
    'reward_shaping': 'none',  # or 'win_bonus', 'action_penalty'
    'opponent_tracking_window': 50,
}

# ============================================================================
# LOGGING & VISUALIZATION
# ============================================================================

LOGGING_CONFIG = {
    'use_wandb': False,  # Set to True to log to Weights & Biases
    'wandb_project': 'chefs-hat-task2',
    'save_plots': True,
    'verbose': 1,  # 0=minimal, 1=normal, 2=verbose
}

# ============================================================================
# HYPERPARAMETER TUNING PRESETS
# ============================================================================

# For faster convergence (fewer timesteps)
FAST_CONFIG = {
    'total_timesteps': 50000,
    'learning_rate': 5e-4,
    'ent_coef': 0.02,
}

# For better exploration (slower but more thorough)
EXPLORATION_CONFIG = {
    'total_timesteps': 200000,
    'learning_rate': 1e-4,
    'ent_coef': 0.05,
    'clip_range': 0.1,
}

# For stability (conservative training)
STABILITY_CONFIG = {
    'total_timesteps': 100000,
    'learning_rate': 1e-4,
    'n_epochs': 20,
    'clip_range': 0.1,
    'ent_coef': 0.001,
}

# ============================================================================
# OPPONENT CONFIGURATION
# ============================================================================

OPPONENT_CONFIG = {
    'random': {
        'description': 'Random action selection',
        'difficulty': 'easy',
    },
    'heuristic': {
        'description': 'Rule-based LargerValue agent',
        'difficulty': 'medium',
    },
    'mixed': {
        'description': '50% random, 50% heuristic',
        'difficulty': 'hard',
    },
    'self': {
        'description': 'Self-play against trained agent',
        'difficulty': 'expert',
    },
}

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

# Which metrics to track and export
TRACKED_METRICS = {
    'training': [
        'timesteps',
        'mean_reward',
        'std_reward',
        'entropy',
        'policy_loss',
        'value_loss',
    ],
    'evaluation': [
        'win_rate',
        'mean_reward',
        'std_reward',
        'mean_episode_length',
        'performance_score',
        'best_10pct_reward',
        'worst_10pct_reward',
    ],
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config(experiment_name: str):
    """Get configuration for a specific experiment."""
    return TRAINING_CONFIG.get(experiment_name, {})

def get_all_experiments():
    """Get list of all configured experiments."""
    return list(TRAINING_CONFIG.keys())

def merge_configs(base_config: dict, overrides: dict):
    """Merge override config into base config."""
    config = base_config.copy()
    config.update(overrides)
    return config

# ============================================================================
# QUICK ACCESS - Copy these to use in your scripts
# ============================================================================

"""
Example usage:

    from config import get_config, EVALUATION_CONFIG
    
    exp_config = get_config('exp1_random')
    trainer = PPOTrainer(
        opponent_type=exp_config['opponent_type'],
    )
    model = trainer.train(**exp_config)
    
    metrics = trainer.evaluate(num_episodes=EVALUATION_CONFIG['num_episodes'])
"""
