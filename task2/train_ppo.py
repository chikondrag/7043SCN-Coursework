"""
PPO Agent Training Script for Chef's Hat Gym

This script trains a PPO agent with opponent modelling capabilities.
Supports training against:
1. Random opponents
2. Rule-based (heuristic) opponents
3. Previously trained agents
4. Self-play scenarios

Task 2, Opponent Modelling Variant (Student ID mod 7 = 1)
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import gymnasium as gym

# Import the ChefsHat environment to register it
try:
    import chefs_env  # This registers ChefsHat-v0
except ImportError:
    pass

# For ChefsHat environment (optional, can use mock for testing)
try:
    import ChefsHatGym
except ImportError:
    pass

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback
)

try:
    import wandb
except ImportError:
    wandb = None

from opponent_modeller import OpponentModeller
from environment_wrapper import ChefsHatWrapper


class OpponentModellingCallback(BaseCallback):
    """
    Custom callback to track opponent modelling during training.
    """
    
    def __init__(self, modeller: OpponentModeller, eval_freq: int = 1000,
                 verbose: int = 0):
        super().__init__(verbose)
        self.modeller = modeller
        self.eval_freq = eval_freq
        self.eval_count = 0
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        if self.n_calls % self.eval_freq == 0:
            # Log opponent modelling insights
            report = self.modeller.get_non_stationarity_report()
            summary = self.modeller.get_summary_stats()
            
            if self.verbose > 0:
                print(f"\n[Step {self.num_timesteps}] Opponent Modelling Report:")
                print(f"  Games recorded: {summary['total_games_recorded']}")
                print(f"  Avg Non-stationarity: {summary['average_nonstationarity']:.3f}")
                for opp_id, stats in report.items():
                    print(f"  Opponent {opp_id}: Type={stats['type']}, "
                          f"WinRate={stats['win_rate']:.2f}, "
                          f"NonStat={stats['nonstationarity_score']:.3f}")
            
            # Log to wandb if available
            try:
                if wandb is not None:
                    wandb.log({
                        "avg_nonstationarity": summary['average_nonstationarity'],
                        "total_games_recorded": summary['total_games_recorded'],
                    }, step=self.num_timesteps)
            except Exception:
                pass
        
        return True


class PPOTrainer:
    """
    Main trainer for PPO agent with opponent modelling.
    """
    
    def __init__(self,
                 opponent_type: str = "random",
                 experiment_name: Optional[str] = None,
                 use_opponent_modelling: bool = True,
                 reward_shaping: str = "none",
                 use_wandb: bool = False):
        """
        Initialize trainer.
        
        Args:
            opponent_type: "random", "heuristic", "self", or "mixed"
            experiment_name: Name for this experiment
            use_opponent_modelling: Whether to track opponent behavior
            reward_shaping: Type of reward shaping to apply
            use_wandb: Whether to log to Weights & Biases
        """
        self.opponent_type = opponent_type
        self.experiment_name = experiment_name or f"ppo_{opponent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.use_opponent_modelling = use_opponent_modelling
        self.reward_shaping = reward_shaping
        self.use_wandb = use_wandb
        
        # Setup directories
        self.model_dir = Path("models") / self.experiment_name
        self.log_dir = Path("logs") / self.experiment_name
        self.result_dir = Path("results") / self.experiment_name
        
        for d in [self.model_dir, self.log_dir, self.result_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.opponent_modeller = OpponentModeller(num_opponents=3, window_size=50)
        self.training_history = {
            'timesteps': [],
            'mean_reward': [],
            'std_reward': [],
        }
        
        # Initialize wandb if enabled
        if self.use_wandb:
            if wandb is None:
                print("Warning: wandb not installed. Install with: pip install wandb")
                self.use_wandb = False
            else:
                try:
                    wandb.init(
                        project="chefs-hat-task2",
                        name=self.experiment_name,
                        config={
                            'opponent_type': opponent_type,
                            'use_opponent_modelling': use_opponent_modelling,
                            'reward_shaping': reward_shaping,
                        }
                    )
                except Exception as e:
                    print(f"Warning: Could not initialize wandb: {e}")
                    self.use_wandb = False
    
    def create_environment(self) -> gym.Env:
        """
        Create training environment.
        """
        try:
            base_env = gym.make("ChefsHat-v0")
        except Exception as e:
            # If ChefsHat-v0 not registered, try to import and register it
            try:
                import sys
                from pathlib import Path
                # Add src to path to import from local ChefsHatGym
                src_path = Path(__file__).parent.parent / "src"
                if src_path.exists() and str(src_path) not in sys.path:
                    sys.path.insert(0, str(src_path))
                
                import ChefsHatGym  # This should register the environment
                base_env = gym.make("ChefsHat-v0")
            except Exception as e2:
                print(f"Error creating ChefsHat environment: {e}")
                print("Make sure chefshatgym is installed via: pip install chefshatgym")
                raise
        
        # Wrap with our wrapper
        env = ChefsHatWrapper(
            base_env,
            opponent_modelling=self.use_opponent_modelling,
            reward_shaping=self.reward_shaping
        )
        
        return env
    
    def configure_opponents(self):
        """
        Configure opponent agents based on opponent_type.
        """
        # This would integrate with existing agents from src/agents/
        # For now, we'll prepare the infrastructure
        
        opponent_config = {
            "random": {
                "agent_type": "RandomAgent",
                "kwargs": {}
            },
            "heuristic": {
                "agent_type": "LargerValueAgent",
                "kwargs": {}
            },
            "self": {
                "agent_type": "PPOAgent",
                "kwargs": {"model_path": str(self.model_dir / "best_model")}
            },
            "mixed": {
                "agents": ["RandomAgent", "LargerValueAgent"],
                "sampling_mode": "random"
            }
        }
        
        return opponent_config.get(self.opponent_type, opponent_config["random"])
    
    def train(self,
              total_timesteps: int = 100000,
              learning_rate: float = 3e-4,
              n_steps: int = 2048,
              batch_size: int = 64,
              n_epochs: int = 10,
              gamma: float = 0.99,
              gae_lambda: float = 0.95,
              clip_range: float = 0.2,
              ent_coef: float = 0.01,
              eval_episodes: int = 10,
              eval_freq: int = 5000,
              save_freq: int = 5000,
              verbose: int = 1):
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total training steps
            learning_rate: Learning rate for PPO optimizer
            n_steps: Rollout buffer size
            batch_size: Batch size for updates
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            eval_episodes: Number of evaluation episodes
            eval_freq: Evaluation frequency
            save_freq: Model saving frequency
            verbose: Verbosity level
        """
        
        print(f"\n{'='*60}")
        print(f"Starting PPO Training")
        print(f"{'='*60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Opponent Type: {self.opponent_type}")
        print(f"Use Opponent Modelling: {self.use_opponent_modelling}")
        print(f"Total Timesteps: {total_timesteps}")
        print(f"Learning Rate: {learning_rate}")
        print(f"{'='*60}\n")
        
        # Create environment
        env = self.create_environment()
        
        # Create PPO model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=verbose,
            tensorboard_log=str(self.log_dir),
        )
        
        # Setup callbacks
        callbacks = [
            OpponentModellingCallback(
                self.opponent_modeller,
                eval_freq=eval_freq,
                verbose=verbose
            ),
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(self.model_dir),
                name_prefix="checkpoint",
                verbose=verbose
            ),
        ]
        
        # Train
        print(f"Training started at {datetime.now()}")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        
        # Save final model
        final_model_path = self.model_dir / "final_model"
        model.save(str(final_model_path))
        print(f"\nFinal model saved to: {final_model_path}")
        
        # Save opponent modelling data
        self._save_opponent_data()
        
        env.close()
        
        return model
    
    def _save_opponent_data(self):
        """Save opponent modelling data and analysis."""
        # Save profiles
        profiles_path = self.result_dir / "opponent_profiles.json"
        self.opponent_modeller.export_profiles(str(profiles_path))
        
        # Save non-stationarity report
        report = self.opponent_modeller.get_non_stationarity_report()
        import json
        with open(self.result_dir / "non_stationarity_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save summary stats
        summary = self.opponent_modeller.get_summary_stats()
        with open(self.result_dir / "summary_stats.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nOpponent data saved to: {self.result_dir}/")
    
    def evaluate(self,
                 model_path: Optional[str] = None,
                 num_episodes: int = 20,
                 verbose: int = 1) -> Dict[str, float]:
        """
        Evaluate trained agent.
        
        Args:
            model_path: Path to saved model (None = use latest)
            num_episodes: Number of evaluation episodes
            verbose: Verbosity level
            
        Returns:
            Evaluation metrics
        """
        
        if model_path is None:
            model_path = str(self.model_dir / "final_model")
        
        # Load model
        try:
            model = PPO.load(model_path)
            print(f"Loaded model from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return {}
        
        # Create evaluation environment
        env = self.create_environment()
        
        # Run evaluation
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if verbose > 0:
                print(f"Episode {episode+1}/{num_episodes}: "
                      f"Reward={episode_reward:.2f}, Length={episode_length}")
        
        # Compute metrics
        metrics = {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_episode_length': float(np.mean(episode_lengths)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
        }
        
        env.close()
        
        print(f"\nEvaluation Results:")
        print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Mean Episode Length: {metrics['mean_episode_length']:.0f}")
        
        return metrics


def main():
    """Main training script."""
    
    # Configuration
    config = {
        'opponent_type': 'random',  # or 'heuristic', 'self', 'mixed'
        'total_timesteps': 100000,
        'learning_rate': 3e-4,
        'save_freq': 5000,
        'eval_freq': 5000,
        'use_opponent_modelling': True,
        'use_wandb': False,
    }
    
    # Create trainer
    trainer = PPOTrainer(
        opponent_type=config['opponent_type'],
        use_opponent_modelling=config['use_opponent_modelling'],
        use_wandb=config['use_wandb'],
    )
    
    # Train
    model = trainer.train(
        total_timesteps=config['total_timesteps'],
        learning_rate=config['learning_rate'],
        save_freq=config['save_freq'],
        eval_freq=config['eval_freq'],
    )
    
    # Evaluate
    metrics = trainer.evaluate(num_episodes=20)
    
    return trainer, model, metrics


if __name__ == "__main__":
    trainer, model, metrics = main()
    print("\nTraining complete!")
    print(f"Experiment: {trainer.experiment_name}")
