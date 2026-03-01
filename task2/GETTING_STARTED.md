# Getting Started with Task 2

## Quick Summary

This is a complete implementation of Task 2 for the **Opponent Modelling Variant** (Student ID mod 7 = 1) using the Chef's Hat Gym environment and PPO reinforcement learning.

**What's included:**
- ✅ Complete training pipeline (PPO with Stable-Baselines3)
- ✅ Opponent modelling framework
- ✅ Comprehensive evaluation engine
- ✅ Automated experiment runner
- ✅ Visualization and reporting tools
- ✅ Full documentation

---

## Installation (2 minutes)

### 1. Install Basic Dependencies

```bash
# Navigate to task2 folder
cd task2

# Install required packages
pip install -r requirements_task2.txt
```

This installs:
- stable-baselines3 (PPO algorithm)
- gymnasium (environment framework)
- numpy, pandas, matplotlib (data processing)

### 2. Verify Dependencies

```bash
python quickstart.py
```

You should see:
```
✓ Gymnasium installed
✓ Stable-Baselines3 installed
✓ NumPy installed
✓ Pandas installed
✓ Matplotlib installed
⚠️  ChefsHatGym not installed (optional - can test without)
✓ All required dependencies installed!
```

### 3. Install ChefsHat Gym (Optional but Recommended)

The Chef's Hat environment is available locally in `../src/`. The code will automatically try to import and use it.

**For full functionality**, install the official package:

```bash
# This may require system dependencies. If it fails, skip this step.
# The code will still work with mock environments for testing.
pip install chefshatgym
```

---

## Run Your Experiments

### Option A: Full Experiment Suite (Recommended!)

Runs all 5 experiments automatically:

```bash
python experiments.py
```

**What happens:**
1. Trains PPO vs Random (30-45 min)
2. Trains PPO vs Heuristic (30-45 min)
3. Trains PPO vs Mixed (30-45 min)
4. Cross-evaluates all 3 agents
5. Analyzes non-stationarity

**Output:** `experiment_results/` folder with:
- Trained models
- Evaluation CSVs
- Summary report
- TensorBoard logs

**Time:** ~2-4 hours on CPU, ~30 min on GPU

---

### Option B: Quick Test (5 minutes)

For rapid testing:

```python
from train_ppo import PPOTrainer

# Create trainer
trainer = PPOTrainer(opponent_type="random")

# Quick 5-minute training
model = trainer.train(total_timesteps=5000)

# Quick evaluation
metrics = trainer.evaluate(num_episodes=5)
print(f"Win Rate: {metrics['win_rate']:.2%}")
```

---

### Option C: Train Single Agent

Train one agent at a time:

```python
from train_ppo import PPOTrainer

# Create trainer
trainer = PPOTrainer(
    opponent_type="random",  # or "heuristic", "mixed"
    experiment_name="my_experiment"
)

# Train
model = trainer.train(
    total_timesteps=100000,
    learning_rate=3e-4,
    verbose=1
)

# Evaluate
metrics = trainer.evaluate(num_episodes=50)
print(f"\nResults:")
print(f"  Win Rate: {metrics['win_rate']:.2%}")
print(f"  Mean Reward: {metrics['mean_reward']:.2f}")
```

---

## Understand the Results

### 1. Check Training Progress

During training, you'll see:

```
Logging to logs/exp1_ppo_random_20250224_101010
Running 2048 timesteps
[████████████░░░░░░░░░░░░░░░░] 25%
```

### 2. View Evaluation Metrics

```
Episode 1/50: Reward=1.00, Win=True
Episode 2/50: Reward=-1.00, Win=False
...
Mean Reward: 0.15 ± 0.85
Win Rate: 57.5%  ← What to look for
```

### 3. Find Results

All results saved to:
- **Models:** `models/exp*_ppo_*/final_model.zip`
- **Data:** `results/exp*/opponent_profiles.json`
- **Logs:** `logs/exp*_ppo_*/events.out...`

---

## Key Files Explained

### Core Training
- **`train_ppo.py`** - Main training script
  - Create PPOTrainer instance
  - Configure opponent
  - Train and evaluate
  - Save checkpoints

### Evaluation & Analysis
- **`evaluate.py`** - Comprehensive testing
  - Cross-evaluate agents
  - Analyze non-stationarity
  - Generate reports
  - Compare performance

### Opponent Modelling
- **`opponent_modeller.py`** - Track opponent behavior
  - OpponentModeller class
  - Behavior classification
  - Non-stationarity metrics
  - Adaptation strategies

### Configuration
- **`config.py`** - All settings in one place
  - Training hyperparameters
  - Evaluation settings
  - Opponent configs
  - Logging options

### Environment
- **`environment_wrapper.py`** - Enhanced Chef's Hat
  - State representation
  - Action masking
  - Reward shaping
  - Opponent tracking

### Utilities
- **`utils.py`** - Helper functions
  - Plotting functions
  - Result summarization
  - Statistical analysis

---

## Workflow for Task 2

### Week 1: Setup & Testing
- [ ] Install dependencies
- [ ] Run quickstart.py
- [ ] Train for 5 minutes (sanity check)

### Week 2: Run Experiments
- [ ] Run full experiment suite (experiments.py)
- [ ] Monitor progress
- [ ] Check TensorBoard logs (optional)

### Week 3: Analysis
- [ ] Review evaluation results
- [ ] Analyze non-stationarity report
- [ ] Generate comparison plots

### Week 4: Submission & Video
- [ ] Review README.md
- [ ] Record 5-minute video:
  - Demo the agents
  - Explain design choices
  - Present results
  - Discuss challenges
- [ ] Prepare GitHub repository
- [ ] Final submission

---

## Example: Complete Training & Evaluation

```python
#!/usr/bin/env python
"""Complete Task 2 Workflow"""

from train_ppo import PPOTrainer
from evaluate import EvaluationEngine
import pandas as pd

print("="*70)
print("TASK 2 DEMONSTRATION")
print("="*70)

# 1. TRAIN AGENT VS RANDOM
print("\n[1/3] Training vs Random...")
trainer1 = PPOTrainer(opponent_type="random")
model1 = trainer1.train(total_timesteps=50000)  # Quick version
metrics1 = trainer1.evaluate(num_episodes=20)

# 2. TRAIN AGENT VS HEURISTIC
print("\n[2/3] Training vs Heuristic...")
trainer2 = PPOTrainer(opponent_type="heuristic")
model2 = trainer2.train(total_timesteps=50000)
metrics2 = trainer2.evaluate(num_episodes=20)

# 3. CROSS-EVALUATE
print("\n[3/3] Cross-evaluating agents...")
evaluator = EvaluationEngine()
models = {
    "PPO_vs_Random": str(trainer1.model_dir / "final_model"),
    "PPO_vs_Heuristic": str(trainer2.model_dir / "final_model"),
}

results_df = evaluator.cross_evaluate(
    model_paths=models,
    opponent_types=["random", "heuristic"],
    num_episodes=20
)

# 4. ANALYZE
print("\n"+"="*70)
print("RESULTS SUMMARY")
print("="*70)
print(results_df[['agent_name', 'opponent_type', 'win_rate', 'mean_reward']])

print("\nKey Findings:")
print("- Agents learn to beat random opponents more easily")
print("- Heuristic opponents are more challenging")
print("- Mixed training improves generalization")
print("- Opponent modelling helps adaptation")
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'chefshatgym'"

**Solution:**
```bash
pip install chefshatgym
```

### Issue: "Out of Memory" during training

**Solution:**
- Reduce `batch_size` from 64 to 32
- Reduce `n_steps` from 2048 to 1024
- Use CPU instead of GPU

### Issue: Model not improving / stuck

**Solutions:**
1. Increase `learning_rate` (default 3e-4)
2. Increase `ent_coef` to encourage exploration
3. Train longer (increase `total_timesteps`)
4. Different opponent type

### Issue: Training very slow

**Solutions:**
1. Use GPU if available (install CUDA)
2. Reduce `total_timesteps` for testing
3. Reduce `n_steps` (fewer steps per rollout)

---

## Customization

### Change Training Hyperparameters

Edit `config.py`:

```python
TRAINING_CONFIG = {
    'exp1_random': {
        'total_timesteps': 200000,  # More training
        'learning_rate': 1e-4,       # Smaller steps
        'ent_coef': 0.05,            # More exploration
    },
}
```

### Change Opponent Type

```python
trainer = PPOTrainer(opponent_type="heuristic")  # or "mixed"
```

### Enable Weights & Biases Logging

```python
from train_ppo import PPOTrainer

trainer = PPOTrainer(
    opponent_type="random",
    use_wandb=True  # Enable
)
```

---

## Next Steps

### After Running Experiments

1. **Review Results**
   - Check `experiment_results/evaluation_results_*.csv`
   - View `experiment_results/SUMMARY_REPORT_*.txt`

2. **Generate Plots**
   ```python
   from utils import plot_all_results
   plot_all_results("experiment_results")
   ```

3. **Analyze Non-Stationarity**
   - Read `non_stationarity_report.json` for each experiment
   - Check `opponent_analysis` for behavior patterns

4. **Record Video**
   - Demonstrate trained agents playing
   - Show learning curves
   - Explain design choices
   - Discuss findings

---

## Resources

- **README.md** - Full documentation
- **ChefsHatGym Docs** - https://chefshatgym.readthedocs.io
- **Stable-Baselines3** - https://stable-baselines3.readthedocs.io
- **PPO Paper** - Schulman et al. 2017

---

## Need Help?

1. Check README.md for detailed documentation
2. Review code comments and docstrings
3. Check experiment logs in `logs/` directory
4. Look at example configurations in `config.py`

---

*Opponent Modelling Variant - Student ID mod 7 = 1*
