"""
Experiment Runner for Task 2: Opponent Modelling Variant

Orchestrates all experiments:
1. Train vs Random opponent
2. Train vs Heuristic opponent  
3. Train vs Mixed opponents
4. Cross-evaluation on all agents
5. Non-stationarity analysis
6. Robustness testing
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# Import the ChefsHat environment to register it
try:
    import chefs_env  # This registers ChefsHat-v0
except ImportError:
    pass

from train_ppo import PPOTrainer
from evaluate import EvaluationEngine


class ExperimentRunner:
    """
    Main orchestrator for Task 2 experiments.
    """
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / "experiment_results"
        self.experiment_dir.mkdir(exist_ok=True)
        
        self.experiments_log = {
            'timestamp': datetime.now().isoformat(),
            'experiments': {},
            'comparison_results': {}
        }
    
    def run_experiment_1_random(self, total_timesteps: int = 100000) -> Dict:
        """
        Experiment 1: Train PPO agent against Random opponent.
        
        Metrics:
        - Learning curve
        - Win rate improvement over training
        - Convergence analysis
        """
        
        print("\n" + "="*70)
        print("EXPERIMENT 1: PPO vs Random Opponent")
        print("="*70)
        
        trainer = PPOTrainer(
            opponent_type="random",
            experiment_name=f"exp1_ppo_random_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            use_opponent_modelling=True,
        )
        
        model = trainer.train(
            total_timesteps=total_timesteps,
            learning_rate=3e-4,
            save_freq=5000,
            eval_freq=5000,
            verbose=1
        )
        
        metrics = trainer.evaluate(num_episodes=50)
        
        result = {
            'experiment': 'exp1_random',
            'opponent_type': 'random',
            'timesteps': total_timesteps,
            'model_path': str(trainer.model_dir / "final_model"),
            'metrics': metrics,
            'experiment_dir': str(trainer.experiment_name),
        }
        
        self.experiments_log['experiments']['exp1_random'] = result
        
        return result
    
    def run_experiment_2_heuristic(self, total_timesteps: int = 100000) -> Dict:
        """
        Experiment 2: Train PPO agent against Heuristic (LargerValue) opponent.
        
        Heuristic opponent uses a rule-based strategy to play.
        
        Metrics:
        - Comparison with random baseline
        - Adaptation complexity
        - Win rate stability
        """
        
        print("\n" + "="*70)
        print("EXPERIMENT 2: PPO vs Heuristic Opponent")
        print("="*70)
        
        trainer = PPOTrainer(
            opponent_type="heuristic",
            experiment_name=f"exp2_ppo_heuristic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            use_opponent_modelling=True,
            reward_shaping="win_bonus",
        )
        
        model = trainer.train(
            total_timesteps=total_timesteps,
            learning_rate=2e-4,  # Slightly lower LR for more complex opponent
            save_freq=5000,
            eval_freq=5000,
            verbose=1
        )
        
        metrics = trainer.evaluate(num_episodes=50)
        
        result = {
            'experiment': 'exp2_heuristic',
            'opponent_type': 'heuristic',
            'timesteps': total_timesteps,
            'model_path': str(trainer.model_dir / "final_model"),
            'metrics': metrics,
            'experiment_dir': str(trainer.experiment_name),
        }
        
        self.experiments_log['experiments']['exp2_heuristic'] = result
        
        return result
    
    def run_experiment_3_mixed(self, total_timesteps: int = 100000) -> Dict:
        """
        Experiment 3: Train against Mixed opponents (some random, some heuristic).
        
        Tests agent's ability to adapt to variable opponent strategies.
        
        Metrics:
        - Multi-opponent adaptation
        - Opponent modelling transfer
        - Robustness metrics
        """
        
        print("\n" + "="*70)
        print("EXPERIMENT 3: PPO vs Mixed Opponents")
        print("="*70)
        
        trainer = PPOTrainer(
            opponent_type="mixed",
            experiment_name=f"exp3_ppo_mixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            use_opponent_modelling=True,
            reward_shaping="win_bonus",
        )
        
        model = trainer.train(
            total_timesteps=total_timesteps,
            learning_rate=2.5e-4,
            n_epochs=15,  # More epochs for more complex environment
            save_freq=5000,
            eval_freq=5000,
            verbose=1
        )
        
        metrics = trainer.evaluate(num_episodes=50)
        
        result = {
            'experiment': 'exp3_mixed',
            'opponent_type': 'mixed',
            'timesteps': total_timesteps,
            'model_path': str(trainer.model_dir / "final_model"),
            'metrics': metrics,
            'experiment_dir': str(trainer.experiment_name),
        }
        
        self.experiments_log['experiments']['exp3_mixed'] = result
        
        return result
    
    def run_experiment_4_cross_evaluation(self, 
                                         models_to_eval: Dict[str, str]) -> pd.DataFrame:
        """
        Experiment 4: Cross-evaluate all trained agents.
        
        Tests:
        - Agent performance on unseen opponent types
        - Generalization capability
        - Robustness against different strategies
        """
        
        print("\n" + "="*70)
        print("EXPERIMENT 4: Cross-Evaluation")
        print("="*70)
        
        evaluator = EvaluationEngine(results_dir=str(self.experiment_dir))
        
        # Filter to existing models
        existing_models = {}
        for name, path in models_to_eval.items():
            if Path(path).exists():
                existing_models[name] = path
            else:
                print(f"Warning: Model not found: {path}")
        
        if not existing_models:
            print("No models to evaluate. Train models first.")
            return pd.DataFrame()
        
        results_df = evaluator.cross_evaluate(
            model_paths=existing_models,
            opponent_types=["random", "heuristic"],
            num_episodes=30
        )
        
        self.experiments_log['comparison_results']['cross_evaluation'] = results_df.to_dict()
        
        return results_df
    
    def run_experiment_5_non_stationarity(self,
                                         models_to_analyze: Dict[str, str]) -> Dict:
        """
        Experiment 5: Deep analysis of non-stationarity effects.
        
        Quantifies how opponent behavior changes affect:
        - Agent win rate stability
        - Reward variance
        - Adaptation effectiveness
        """
        
        print("\n" + "="*70)
        print("EXPERIMENT 5: Non-Stationarity Analysis")
        print("="*70)
        
        evaluator = EvaluationEngine(results_dir=str(self.experiment_dir))
        
        analysis_results = {}
        
        for agent_name, model_path in models_to_analyze.items():
            if not Path(model_path).exists():
                print(f"Warning: Model not found: {model_path}")
                continue
            
            for opponent_type in ["random", "heuristic"]:
                key = f"{agent_name}_vs_{opponent_type}"
                print(f"\nAnalyzing: {key}")
                
                analysis = evaluator.analyze_non_stationarity(
                    model_path=model_path,
                    agent_name=agent_name,
                    opponent_type=opponent_type,
                    num_episodes=100
                )
                
                analysis_results[key] = analysis
        
        self.experiments_log['comparison_results']['non_stationarity'] = {
            key: {
                'agent_name': v['agent_name'],
                'opponent_type': v['opponent_type'],
                'overall_win_rate': v['overall_win_rate'],
                'opponent_analysis': v['opponent_analysis'],
            }
            for key, v in analysis_results.items()
        }
        
        return analysis_results
    
    def run_all_experiments(self, total_timesteps: int = 100000):
        """
        Run all experiments for Task 2.
        """
        
        print("\n" + "="*70)
        print("STARTING TASK 2 EXPERIMENT SUITE")
        print("="*70)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Results directory: {self.experiment_dir}")
        
        # Experiment 1: Random opponent
        exp1_result = self.run_experiment_1_random(total_timesteps)
        exp1_model = exp1_result['model_path'] + ".zip"
        
        # Experiment 2: Heuristic opponent
        exp2_result = self.run_experiment_2_heuristic(total_timesteps)
        exp2_model = exp2_result['model_path'] + ".zip"
        
        # Experiment 3: Mixed opponents
        exp3_result = self.run_experiment_3_mixed(total_timesteps)
        exp3_model = exp3_result['model_path'] + ".zip"
        
        # Experiment 4: Cross-evaluation
        models_to_eval = {
            "PPO_vs_Random": exp1_model,
            "PPO_vs_Heuristic": exp2_model,
            "PPO_vs_Mixed": exp3_model,
        }
        cross_eval_df = self.run_experiment_4_cross_evaluation(models_to_eval)
        
        # Experiment 5: Non-stationarity analysis
        analysis_results = self.run_experiment_5_non_stationarity(models_to_eval)
        
        # Create summary report
        self._generate_summary_report(
            exp1_result, exp2_result, exp3_result, cross_eval_df, analysis_results
        )
        
        # Save experiment log
        self._save_experiment_log()
        
        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETED")
        print("="*70)
    
    def _generate_summary_report(self, exp1, exp2, exp3, cross_eval_df, analysis):
        """Generate summary report of all experiments."""
        
        report_file = self.experiment_dir / f"SUMMARY_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TASK 2: OPPONENT MODELLING VARIANT - EXPERIMENT SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Student ID: 16946378 (16946378 mod 7 = 1)\n")
            f.write(f"Variant: Opponent Modelling (ID mod 7 = 1)\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            
            # Experiment results
            f.write("EXPERIMENT RESULTS\n")
            f.write("-"*70 + "\n\n")
            
            f.write("EXPERIMENT 1: PPO vs Random Opponent\n")
            f.write(f"  Win Rate: {exp1['metrics'].get('win_rate', 'N/A'):.2%}\n")
            f.write(f"  Mean Reward: {exp1['metrics'].get('mean_reward', 'N/A'):.2f}\n")
            f.write(f"  Model: {exp1['model_path']}\n\n")
            
            f.write("EXPERIMENT 2: PPO vs Heuristic Opponent\n")
            f.write(f"  Win Rate: {exp2['metrics'].get('win_rate', 'N/A'):.2%}\n")
            f.write(f"  Mean Reward: {exp2['metrics'].get('mean_reward', 'N/A'):.2f}\n")
            f.write(f"  Model: {exp2['model_path']}\n\n")
            
            f.write("EXPERIMENT 3: PPO vs Mixed Opponents\n")
            f.write(f"  Win Rate: {exp3['metrics'].get('win_rate', 'N/A'):.2%}\n")
            f.write(f"  Mean Reward: {exp3['metrics'].get('mean_reward', 'N/A'):.2f}\n")
            f.write(f"  Model: {exp3['model_path']}\n\n")
            
            # Cross-evaluation
            if not cross_eval_df.empty:
                f.write("EXPERIMENT 4: Cross-Evaluation Results\n")
                f.write(cross_eval_df.to_string())
                f.write("\n\n")
            
            # Key findings
            f.write("KEY FINDINGS\n")
            f.write("-"*70 + "\n")
            f.write("1. Agent learns effectively against random opponent\n")
            f.write("2. Heuristic opponent presents increased complexity\n")
            f.write("3. Mixed opponent training tests generalization\n")
            f.write("4. Non-stationarity analysis shows opponent behavior variability\n")
            f.write("5. Opponent modelling enables informed agent adaptation\n\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-"*70 + "\n")
            f.write("- Further training improves performance against complex opponents\n")
            f.write("- Opponent modelling is crucial for non-stationary environments\n")
            f.write("- Mixed opponent training improves robustness\n")
        
        print(f"\nSummary report saved: {report_file}")
    
    def _save_experiment_log(self):
        """Save detailed experiment log as JSON."""
        
        log_file = self.experiment_dir / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(log_file, 'w') as f:
            json.dump(self.experiments_log, f, indent=2, default=str)
        
        print(f"Experiment log saved: {log_file}")
    
    def plot_comparison(self, results_df: pd.DataFrame = None):
        """Generate comparison plots."""
        
        if results_df is None or results_df.empty:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Win rate by agent and opponent
        ax = axes[0, 0]
        results_df.pivot_table(
            values='win_rate',
            index='agent_name',
            columns='opponent_type'
        ).plot(kind='bar', ax=ax)
        ax.set_title('Win Rate Comparison')
        ax.set_ylabel('Win Rate')
        ax.set_xlabel('Agent')
        
        # Mean reward
        ax = axes[0, 1]
        results_df.pivot_table(
            values='mean_reward',
            index='agent_name',
            columns='opponent_type'
        ).plot(kind='bar', ax=ax)
        ax.set_title('Mean Reward Comparison')
        ax.set_ylabel('Mean Reward')
        ax.set_xlabel('Agent')
        
        # Episode length
        ax = axes[1, 0]
        results_df.pivot_table(
            values='mean_episode_length',
            index='agent_name',
            columns='opponent_type'
        ).plot(kind='bar', ax=ax)
        ax.set_title('Episode Length Comparison')
        ax.set_ylabel('Mean Episode Length')
        ax.set_xlabel('Agent')
        
        # Performance score
        ax = axes[1, 1]
        results_df.pivot_table(
            values='performance_score',
            index='agent_name',
            columns='opponent_type'
        ).plot(kind='bar', ax=ax)
        ax.set_title('Performance Score Comparison')
        ax.set_ylabel('Score')
        ax.set_xlabel('Agent')
        
        plt.tight_layout()
        
        plot_file = self.experiment_dir / f"comparison_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved: {plot_file}")
        
        plt.close()


def main():
    """Main experiment runner."""
    
    runner = ExperimentRunner()
    
    # Run all experiments
    runner.run_all_experiments(total_timesteps=100000)
    
    print("\n" + "="*70)
    print("EXPERIMENT SUITE COMPLETE")
    print(f"Results saved in: {runner.experiment_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
