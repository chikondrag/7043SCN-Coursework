"""
Utilities and Plotting Functions for Task 2

Helper functions for:
- Plotting learning curves
- Comparing results
- Analyzing non-stationarity
- Generating reports
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import json

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_learning_curves(results_dir: str, save_path: Optional[str] = None):
    """
    Plot learning curves from TensorBoard logs or metrics files.
    
    Args:
        results_dir: Directory containing experiment results
        save_path: Where to save the figure
    """
    results_dir = Path(results_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Try to read metrics from JSON files
    metrics_files = list(results_dir.glob('*/metrics.json'))
    
    if not metrics_files:
        print(f"No metrics files found in {results_dir}")
        return
    
    for ax, metrics_file in zip(axes.flat, metrics_files[:4]):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            if isinstance(metrics, dict) and 'timesteps' in metrics:
                ax.plot(metrics['timesteps'], metrics['mean_reward'], 
                       label='Mean Reward', linewidth=2)
                ax.fill_between(metrics['timesteps'], 
                               np.array(metrics['mean_reward']) - np.array(metrics['std_reward']),
                               np.array(metrics['mean_reward']) + np.array(metrics['std_reward']),
                               alpha=0.3)
                ax.set_xlabel('Timesteps')
                ax.set_ylabel('Mean Reward')
                ax.set_title(f"Learning Curve - {metrics_file.parent.name}")
                ax.legend()
                ax.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to: {save_path}")
    
    return fig


def plot_win_rate_comparison(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot win rate comparison across agents and opponent types.
    
    Args:
        df: DataFrame with evaluation results
        save_path: Where to save the figure
    """
    if df.empty:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Win rates by agent
    ax = axes[0]
    win_rates = df.groupby('agent_name')['win_rate'].mean()
    colors = plt.cm.Set3(np.linspace(0, 1, len(win_rates)))
    win_rates.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title('Average Win Rate by Agent', fontsize=14, fontweight='bold')
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.set_xlabel('Agent', fontsize=12)
    ax.set_ylim([0, 1])
    ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Random Baseline (4 players)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Win rates by opponent type
    ax = axes[1]
    pivot_df = df.pivot_table(values='win_rate', index='agent_name', columns='opponent_type')
    pivot_df.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'], 
                  edgecolor='black', linewidth=1.5)
    ax.set_title('Win Rate by Opponent Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.set_xlabel('Agent', fontsize=12)
    ax.set_ylim([0, 1])
    ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5)
    ax.legend(title='Opponent Type')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Win rate comparison saved to: {save_path}")
    
    return fig


def plot_non_stationarity_analysis(analysis: Dict, save_path: Optional[str] = None):
    """
    Plot non-stationarity analysis results.
    
    Args:
        analysis: Dictionary with non-stationarity analysis
        save_path: Where to save the figure
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    agents = list(analysis.keys())
    nonstationarity_scores = []
    consistency_scores = []
    
    for agent_analysis in analysis.values():
        if 'opponent_analysis' in agent_analysis:
            opp_analysis = agent_analysis['opponent_analysis']
            # Average across opponents
            scores = [opp['nonstationarity_score'] for opp in opp_analysis.values()]
            consistency = [opp['consistency_score'] for opp in opp_analysis.values()]
            nonstationarity_scores.append(np.mean(scores))
            consistency_scores.append(np.mean(consistency))
    
    # Plot 1: Non-stationarity scores
    ax = axes[0]
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(agents)))
    bars1 = ax.bar(range(len(agents)), nonstationarity_scores, color=colors, 
                   edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.set_ylabel('Non-Stationarity Score', fontsize=12)
    ax.set_title('Opponent Non-Stationarity by Agent', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars1, nonstationarity_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Consistency scores (complement of non-stationarity)
    ax = axes[1]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(agents)))
    bars2 = ax.bar(range(len(agents)), consistency_scores, color=colors,
                   edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.set_ylabel('Consistency Score', fontsize=12)
    ax.set_title('Opponent Consistency by Agent', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars2, consistency_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Non-stationarity analysis plot saved to: {save_path}")
    
    return fig


def plot_performance_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Box plot showing performance distribution across episodes.
    
    Args:
        df: DataFrame with evaluation results
        save_path: Where to save the figure
    """
    
    if df.empty or 'agent_name' not in df.columns:
        print("No data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create box plot for each agent
    agents = df['agent_name'].unique()
    data_to_plot = [df[df['agent_name'] == agent]['mean_reward'].values for agent in agents]
    
    bp = ax.boxplot(data_to_plot, labels=agents, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(agents)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Performance Distribution by Agent', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance distribution plot saved to: {save_path}")
    
    return fig


def create_results_summary(results_dir: str, output_file: str = "RESULTS_SUMMARY.txt"):
    """
    Create a text summary of all results.
    
    Args:
        results_dir: Directory containing results
        output_file: Where to save the summary
    """
    results_dir = Path(results_dir)
    output_file = results_dir / output_file
    
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TASK 2: OPPONENT MODELLING VARIANT - RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        # Look for evaluation CSV files
        csv_files = list(results_dir.glob("evaluation_results_*.csv"))
        
        for csv_file in csv_files:
            f.write(f"\nFile: {csv_file.name}\n")
            f.write("-"*70 + "\n")
            
            df = pd.read_csv(csv_file)
            
            # Summary by agent
            for agent in df['agent_name'].unique():
                agent_df = df[df['agent_name'] == agent]
                f.write(f"\nAgent: {agent}\n")
                f.write(f"  Average Win Rate: {agent_df['win_rate'].mean():.2%}\n")
                f.write(f"  Average Reward: {agent_df['mean_reward'].mean():.2f}\n")
                f.write(f"  Average Episode Length: {agent_df['mean_episode_length'].mean():.0f}\n")
                f.write(f"  Performance Score: {agent_df['performance_score'].mean():.2f}\n")
        
        # Non-stationarity report
        report_files = list(results_dir.glob("*/non_stationarity_report.json"))
        if report_files:
            f.write(f"\n\nNON-STATIONARITY ANALYSIS\n")
            f.write("-"*70 + "\n")
            
            for report_file in report_files:
                with open(report_file, 'r') as rf:
                    report = json.load(rf)
                
                f.write(f"\nExperiment: {report_file.parent.name}\n")
                for opp_id, stats in report.items():
                    f.write(f"  Opponent {opp_id}:\n")
                    f.write(f"    Type: {stats['type']}\n")
                    f.write(f"    Non-Stationarity: {stats['nonstationarity_score']:.3f}\n")
                    f.write(f"    Win Rate: {stats['win_rate']:.2%}\n")
    
    print(f"Summary saved to: {output_file}")


def compare_opponents(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comparison table for opponent types.
    
    Args:
        results_df: DataFrame with evaluation results
        
    Returns:
        Comparison DataFrame
    """
    if results_df.empty:
        return pd.DataFrame()
    
    comparison = results_df.groupby('opponent_type').agg({
        'win_rate': ['mean', 'std'],
        'mean_reward': ['mean', 'std'],
        'mean_episode_length': ['mean', 'std'],
        'performance_score': ['mean', 'std'],
    })
    
    return comparison


def print_experiment_summary(trainer_list: List, evaluator_results: Dict):
    """
    Print a formatted summary of experiments.
    
    Args:
        trainer_list: List of trained model objects
        evaluator_results: Dictionary of evaluation results
    """
    
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70 + "\n")
    
    print(f"Models Trained: {len(trainer_list)}")
    print(f"Evaluations Completed: {len(evaluator_results)}\n")
    
    print("Key Metrics:")
    print("-"*70)
    
    for exp_name, results in evaluator_results.items():
        if isinstance(results, dict) and 'metrics' in results:
            metrics = results['metrics']
            print(f"\n{exp_name}:")
            print(f"  Win Rate: {metrics.get('win_rate', 'N/A'):.2%}")
            print(f"  Mean Reward: {metrics.get('mean_reward', 'N/A'):.2f}")
            print(f"  Mean Episode Length: {metrics.get('mean_episode_length', 'N/A'):.0f}")


# Convenience function for quick plotting
def plot_all_results(results_dir: str):
    """Generate all standard plots."""
    results_dir = Path(results_dir)
    
    print(f"Generating plots from {results_dir}...\n")
    
    # Read evaluation results
    csv_files = list(results_dir.glob("evaluation_results_*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[0])
        
        plot_win_rate_comparison(df, save_path=results_dir / "win_rate_comparison.png")
        plot_performance_distribution(df, save_path=results_dir / "performance_distribution.png")
    
    # Create summary
    create_results_summary(str(results_dir))
    
    print("\nAll plots generated successfully!")
