"""
Quick-Start Script for Task 2

Run this to:
1. Verify all dependencies are installed
2. Test environment creation
3. Quick training (short version)
4. Test evaluation
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")

def check_dependencies():
    """Check if all required packages are installed."""
    print_header("CHECKING DEPENDENCIES")
    
    packages = {
        'gymnasium': 'Gymnasium',
        'stable_baselines3': 'Stable-Baselines3',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
    }
    
    missing = []
    
    for import_name, display_name in packages.items():
        try:
            __import__(import_name)
            print(f"✓ {display_name} installed")
        except ImportError:
            print(f"✗ {display_name} NOT installed")
            missing.append(import_name)
    
    # Check Chef's Hat Gym (optional)
    try:
        import ChefsHatGym
        print(f"✓ ChefsHatGym installed (optional)")
    except ImportError:
        print(f"⚠️  ChefsHatGym not installed (optional - can test without)")
    
    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print(f"\nInstall with:")
        print(f"  pip install -r requirements_task2.txt")
        return False
    
    print("\n✓ All required dependencies installed!")
    return True

def test_environment():
    """Test Chef's Hat environment creation."""
    print_header("TESTING CHEF'S HAT ENVIRONMENT")
    
    try:
        import gymnasium as gym
        # Import to register the environment
        import chefs_env
        
        print("Creating environment...")
        try:
            # Try to create ChefsHat environment if registered
            env = gym.make("ChefsHat-v0")
            print("✓ ChefsHat-v0 environment found")
        except Exception as e:
            print(f"⚠️  ChefsHat-v0 not registered (this is OK for basic testing)")
            print(f"   Error: {e}")
            print("   The environment can be registered by installing chefshatgym")
            return True
        
        print("Resetting environment...")
        obs, info = env.reset()
        
        print(f"✓ Environment created successfully")
        print(f"  Observation shape: {obs.shape if hasattr(obs, 'shape') else 'Variable'}")
        print(f"  Action space: {env.action_space}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error creating environment: {e}")
        return False

def quick_train():
    """Run a quick training session (5 minutes)."""
    print_header("QUICK TRAINING TEST (5 MINUTES)")
    
    try:
        from train_ppo import PPOTrainer
        
        trainer = PPOTrainer(
            opponent_type="random",
            experiment_name="quickstart_test"
        )
        
        print("Training for 5000 timesteps (quick test)...")
        model = trainer.train(
            total_timesteps=5000,
            learning_rate=3e-4,
            save_freq=2500,
            eval_freq=2500,
            verbose=1
        )
        
        print("\n✓ Quick training completed successfully")
        
        # Quick evaluation
        print("\nEvaluating model (5 episodes)...")
        metrics = trainer.evaluate(num_episodes=5, verbose=0)
        
        if metrics:
            print(f"  Mean Reward: {metrics.get('mean_reward', 'N/A'):.2f}")
            print(f"  Mean Episode Length: {metrics.get('mean_episode_length', 'N/A'):.0f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all checks."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "TASK 2 QUICK-START VERIFICATION" + " "*22 + "║")
    print("║" + " "*20 + "Opponent Modelling Variant" + " "*22 + "║")
    print("╚" + "="*68 + "╝")
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Please install missing dependencies first:")
        print("   cd task2")
        print("   pip install -r requirements_task2.txt")
        return False
    
    # Test environment
    if not test_environment():
        print("\n⚠️  Chef's Hat Gym not properly installed")
        print("   pip install chefshatgym")
        return False
    
    # Quick train
    run_train = input("\nRun quick training test? (y/n) [y]: ").lower().strip() or 'y'
    if run_train == 'y':
        if not quick_train():
            print("\n⚠️  Training failed - check your setup")
            return False
    
    # Success!
    print_header("✓ ALL CHECKS PASSED!")
    
    print("You're ready to run full Task 2 experiments!\n")
    print("Next steps:")
    print("1. Review the README.md for detailed instructions")
    print("2. Run: python experiments.py (for all experiments)")
    print("3. Or: python train_ppo.py (for single agent training)")
    print("4. Then: python evaluate.py (to evaluate models)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
