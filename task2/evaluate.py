"""
Evaluation Script for Task 2

Comprehensive evaluation of trained PPO agents against:
1. Random opponents
2. Heuristic opponents
3. Other trained agents (cross-evaluation)
4. Analysis of non-stationarity

Generates:
- Win rate statistics
- Performance scores
- Robustness metrics
- Learning curves
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import matplotlib.pyplot as plt
import gymnasium as gym

# Import the ChefsHat environment to register it
try:
    import chefs_env  # This registers ChefsHat-v0
except ImportError:
    pass

# For ChefsHat environment (optional)
try:
    import ChefsHatGym
except ImportError:
    pass

from stable_baselines3 import PPO
from opponent_modeller import OpponentModeller
from environment_wrapper import ChefsHatWrapper


class EvaluationEngine:
    """
    Comprehensive evaluation framework for RL agents in Chef's Hat.
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.evaluation_history = []
    
    def evaluate_agent(self,
                      model_path: str,
                      agent_name: str,
                      opponent_type: str = "random",
                      num_episodes: int = 50,
                      seed: int = 42,
                      verbose: int = 1) -> Dict:
        """
        Evaluate an agent against specific opponent type.
        
        Args:
            model_path: Path to trained PPO model
            agent_name: Name identifier for agent
            opponent_type: Type of opponent ("random", "heuristic", etc.)
            num_episodes: Number of evaluation episodes
            seed: Random seed for reproducibility
            verbose: Verbosity level
            
        Returns:
            Evaluation metrics dictionary
        """
        
        np.random.seed(seed)
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {agent_name} vs {opponent_type}")
        print(f"Episodes: {num_episodes}")
        print(f"{'='*60}")
        
        # Load model
        try:
            model = PPO.load(model_path)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return {}
        
        # Create environment
        env = self._create_environment(opponent_type)
        modeller = OpponentModeller()
        
        # Run evaluation episodes
        episode_data = {
            'rewards': [],
            'lengths': [],
            'wins': [],
            'opponent_indices': [],
        }
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            
            modeller.start_game()
            episode_reward = 0
            episode_length = 0
            done = False
            actions_taken = []
            
            while not done:
                # Agent prediction
                action, _ = model.predict(obs, deterministic=True)
                actions_taken.append(action)
                
                # Environment step
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
                # Track opponent actions if available
                if 'opponent_actions' in info:
                    for opp_idx, opp_action in enumerate(info['opponent_actions']):
                        modeller.record_action(opp_idx, opp_action)
            
            # Record episode results
            is_win = episode_reward > 0
            episode_data['rewards'].append(episode_reward)
            episode_data['lengths'].append(episode_length)
            episode_data['wins'].append(1 if is_win else 0)
            episode_data['opponent_indices'].append(opponent_type)
            
            # End episode in modeller
            final_ranks = {0: 1 if is_win else 4}  # Simplified ranking
            modeller.end_game(final_ranks)
            
            if verbose > 0 and (episode + 1) % 10 == 0:
                print(f"  Episode {episode+1}/{num_episodes} - "
                      f"Reward: {episode_reward:.2f}, Win: {is_win}")
        
        # Compute metrics
        metrics = self._compute_metrics(episode_data, agent_name, opponent_type)
        
        # Add opponent modelling analysis
        metrics['opponent_analysis'] = modeller.get_non_stationarity_report()
        metrics['opponent_summary'] = modeller.get_summary_stats()
        
        env.close()
        
        return metrics
    
    def _create_environment(self, opponent_type: str) -> gym.Env:
        """Create environment with specified opponent type."""
        try:
            base_env = gym.make("ChefsHat-v0")
            env = ChefsHatWrapper(base_env, opponent_modelling=True)
            return env
        except Exception as e:
            print(f"Error creating environment: {e}")
            raise
    
    def _compute_metrics(self, data: Dict, agent_name: str, 
                        opponent_type: str) -> Dict:
        """Compute comprehensive evaluation metrics."""
        
        rewards = np.array(data['rewards'])
        lengths = np.array(data['lengths'])
        wins = np.array(data['wins'])
        
        metrics = {
            'agent_name': agent_name,
            'opponent_type': opponent_type,
            'evaluation_date': datetime.now().isoformat(),
            
            # Reward statistics
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'median_reward': float(np.median(rewards)),
            
            # Win statistics
            'win_rate': float(np.mean(wins)),
            'total_wins': int(np.sum(wins)),
            'total_episodes': len(wins),
            
            # Episode length statistics
            'mean_episode_length': float(np.mean(lengths)),
            'std_episode_length': float(np.std(lengths)),
            'min_episode_length': int(np.min(lengths)),
            'max_episode_length': int(np.max(lengths)),
            
            # Performance score (higher = better)
            'performance_score': float(np.mean(wins) * 100),
        }
        
        # Add tail metrics (worst 10% vs best 10%)
        sorted_rewards = np.sort(rewards)
        tail_idx = len(sorted_rewards) // 10
        metrics['best_10pct_reward'] = float(np.mean(sorted_rewards[-tail_idx:]))
        metrics['worst_10pct_reward'] = float(np.mean(sorted_rewards[:tail_idx]))
        
        return metrics
    
    def cross_evaluate(self,
                      model_paths: Dict[str, str],
                      opponent_types: List[str] = None,
                      num_episodes: int = 30,
                      seed: int = 42) -> pd.DataFrame:
        """
        Cross-evaluate multiple agents against multiple opponent types.
        
        Creates a comprehensive comparison table.
        
        Args:
            model_paths: Dict mapping agent names to model paths
            opponent_types: List of opponent types to test against
            num_episodes: Number of episodes per evaluation
            seed: Random seed
            
        Returns:
            DataFrame with evaluation results
        """
        
        if opponent_types is None:
            opponent_types = ["random", "heuristic"]
        
        results = []
        
        for agent_name, model_path in model_paths.items():
            for opponent_type in opponent_types:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Evaluating {agent_name} vs {opponent_type}...")
                
                metrics = self.evaluate_agent(
                    model_path=model_path,
                    agent_name=agent_name,
                    opponent_type=opponent_type,
                    num_episodes=num_episodes,
                    seed=seed,
                    verbose=0
                )
                
                if metrics:
                    results.append(metrics)
                    print(f"  Win Rate: {metrics['win_rate']:.2%}")
                    print(f"  Mean Reward: {metrics['mean_reward']:.2f}")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        csv_path = self.results_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        return df
    
    def analyze_non_stationarity(self,
                                model_path: str,
                                agent_name: str = "Agent",
                                opponent_type: str = "random",
                                num_episodes: int = 100) -> Dict:
        """
        Deep analysis of non-stationarity effects on agent performance.
        
        Tracks how agent performance changes as opponent behavior changes.
        """
        
        model = PPO.load(model_path)
        env = self._create_environment(opponent_type)
        modeller = OpponentModeller()
        
        # Track performance over time
        sliding_win_rate = []
        window_size = 10
        
        print(f"\n{'='*60}")
        print(f"Non-Stationarity Analysis: {agent_name} vs {opponent_type}")
        print(f"Episodes: {num_episodes}")
        print(f"{'='*60}")
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            modeller.start_game()
            
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            is_win = episode_reward > 0
            modeller.end_game({0: 1 if is_win else 4})
            
            # Compute sliding win rate
            if episode >= window_size:
                recent_games = modeller.game_history[-window_size:]
                recent_wins = sum(1 for g in recent_games if g['final_ranks'].get(0, 4) == 1)
                sliding_win_rate.append(recent_wins / window_size)
        
        # Analysis
        curves = modeller.get_learning_curves()
        report = modeller.get_non_stationarity_report()
        
        analysis = {
            'agent_name': agent_name,
            'opponent_type': opponent_type,
            'total_episodes': num_episodes,
            'overall_win_rate': curves[0][-1] if curves[0] else 0,
            'sliding_win_rates': sliding_win_rate,
            'opponent_analysis': report,
        }
        
        env.close()
        
        # Print analysis
        print(f"\nOpponent Non-Stationarity Report:")
        for opp_id, stats in report.items():
            print(f"  Opponent {opp_id}:")
            print(f"    Type: {stats['type']}")
            print(f"    Non-Stationarity Score: {stats['nonstationarity_score']:.3f}")
            print(f"    Win Rate: {stats['win_rate']:.2%}")
            print(f"    Trend: {stats['trend']:+.3f}")
        
        return analysis
    
    def generate_report(self, results_df: pd.DataFrame, output_file: str = None):
        """Generate comprehensive evaluation report."""
        
        if output_file is None:
            output_file = self.results_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("AGENT EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-"*70 + "\n")
            
            for agent in results_df['agent_name'].unique():
                agent_results = results_df[results_df['agent_name'] == agent]
                f.write(f"\n{agent}:\n")
                f.write(f"  Mean Win Rate: {agent_results['win_rate'].mean():.2%}\n")
                f.write(f"  Mean Performance Score: {agent_results['performance_score'].mean():.2f}\n")
                f.write(f"  Best Performance: {agent_results['performance_score'].max():.2f}\n")
            
            # Detailed results
            f.write("\n\nDETAILED RESULTS\n")
            f.write("-"*70 + "\n")
            f.write(results_df.to_string())
        
        print(f"\nReport saved to: {output_file}")
        return output_file


def main():
    """Main evaluation script."""
    
    evaluator = EvaluationEngine(results_dir="results")
    
    # Example: Cross-evaluate multiple trained models
    model_paths = {
        "PPO_vs_Random": "models/ppo_random_20250224_120000/final_model.zip",
        "PPO_vs_Heuristic": "models/ppo_heuristic_20250224_120000/final_model.zip",
    }
    
    # Run evaluation if models exist
    existing_models = {
        name: path for name, path in model_paths.items()
        if Path(path).exists()
    }
    
    if existing_models:
        results_df = evaluator.cross_evaluate(
            model_paths=existing_models,
            opponent_types=["random", "heuristic"],
            num_episodes=30
        )
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(results_df[['agent_name', 'opponent_type', 'win_rate', 'mean_reward']])
        
        evaluator.generate_report(results_df)
    else:
        print("No trained models found. Train models first using train_ppo.py")


if __name__ == "__main__":
    main()
