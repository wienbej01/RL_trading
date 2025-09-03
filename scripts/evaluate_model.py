import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import Settings
from src.rl.train import build_env
from src.rl.evaluate import ModelEvaluator
from stable_baselines3.common.vec_env import DummyVecEnv

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL model.")
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--config", default="configs/settings.yaml", help="Path to the configuration file")
    parser.add_argument("--data", required=True, help="Path to the evaluation data")
    parser.add_argument("--features", required=True, help="Path to the features for the evaluation data")
    parser.add_argument("--output", default="evaluation", help="Directory to save the evaluation report")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    args = parser.parse_args()

    # Load settings
    settings = Settings.from_paths(args.config)

    # Build environment
    env = build_env(settings, args.data, args.features)
    vec_env = DummyVecEnv([lambda: env])

    # Evaluate model
    evaluator = ModelEvaluator(settings)
    results = evaluator.evaluate_model(args.model, vec_env, num_episodes=args.episodes)
    evaluator.generate_report(args.output)

if __name__ == "__main__":
    main()
