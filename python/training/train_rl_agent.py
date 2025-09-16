#!/usr/bin/env python3
"""
TrackPilot Reinforcement Learning Training Script

This script trains an RL agent for railway traffic scheduling using various algorithms
including PPO, A2C, DQN, and custom policy gradient methods.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    StopTrainingOnRewardThreshold,
    CallbackList,
    CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import wandb
from loguru import logger
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from models.railway_env import RailwaySchedulingEnv
from models.networks import (
    SchedulingActorCritic, 
    DQNNetwork, 
    TransformerScheduler
)
from utils.config import load_config, validate_config
from utils.metrics import TrainingMetrics
from utils.export import export_to_torchscript


class TrainingConfig:
    """Training configuration class"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.algorithm = config_dict.get('algorithm', 'PPO')
        self.total_timesteps = config_dict.get('total_timesteps', 1000000)
        self.learning_rate = config_dict.get('learning_rate', 3e-4)
        self.batch_size = config_dict.get('batch_size', 64)
        self.n_steps = config_dict.get('n_steps', 2048)
        self.gamma = config_dict.get('gamma', 0.99)
        self.gae_lambda = config_dict.get('gae_lambda', 0.95)
        self.clip_range = config_dict.get('clip_range', 0.2)
        self.ent_coef = config_dict.get('ent_coef', 0.01)
        self.vf_coef = config_dict.get('vf_coef', 0.5)
        self.max_grad_norm = config_dict.get('max_grad_norm', 0.5)
        
        # Environment settings
        self.env_config = config_dict.get('environment', {})
        self.n_envs = config_dict.get('n_envs', 4)
        
        # Training settings
        self.save_freq = config_dict.get('save_freq', 10000)
        self.eval_freq = config_dict.get('eval_freq', 5000)
        self.eval_episodes = config_dict.get('eval_episodes', 10)
        
        # Logging
        self.use_wandb = config_dict.get('use_wandb', False)
        self.wandb_project = config_dict.get('wandb_project', 'trackpilot-rl')
        self.log_dir = config_dict.get('log_dir', 'logs')


class CustomCallback(CallbackList):
    """Custom callback for training monitoring and early stopping"""
    
    def __init__(self, config: TrainingConfig, save_path: str):
        self.config = config
        self.save_path = save_path
        self.metrics = TrainingMetrics()
        
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=config.save_freq,
            save_path=save_path,
            name_prefix='trackpilot_model'
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_env = make_vec_env(
            RailwaySchedulingEnv,
            n_envs=1,
            env_kwargs=config.env_config
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=config.eval_freq,
            n_eval_episodes=config.eval_episodes,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Early stopping callback
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=0.95,  # Stop when average reward reaches 95%
            verbose=1
        )
        callbacks.append(stop_callback)
        
        super().__init__(callbacks)
    
    def _on_step(self) -> bool:
        """Called after each training step"""
        # Update custom metrics
        self.metrics.update(
            step=self.num_timesteps,
            episode_reward=self.locals.get('episode_reward', 0),
            episode_length=self.locals.get('episode_length', 0)
        )
        
        # Log to wandb if enabled
        if self.config.use_wandb and self.num_timesteps % 1000 == 0:
            wandb.log(self.metrics.get_current_metrics(), step=self.num_timesteps)
        
        return super()._on_step()


def create_environment(config: TrainingConfig, seed: Optional[int] = None) -> gym.Env:
    """Create training environment with proper configuration"""
    
    def make_env():
        env = RailwaySchedulingEnv(**config.env_config)
        env = Monitor(env)
        if seed is not None:
            env.seed(seed)
        return env
    
    if config.n_envs == 1:
        return DummyVecEnv([make_env])
    else:
        return SubprocVecEnv([make_env for _ in range(config.n_envs)])


def create_model(algorithm: str, env: gym.Env, config: TrainingConfig) -> Any:
    """Create RL model based on algorithm choice"""
    
    common_kwargs = {
        'env': env,
        'learning_rate': config.learning_rate,
        'verbose': 1,
        'tensorboard_log': config.log_dir,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if algorithm == 'PPO':
        return PPO(
            policy='MultiInputPolicy',
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            **common_kwargs
        )
    
    elif algorithm == 'A2C':
        return A2C(
            policy='MultiInputPolicy',
            n_steps=config.n_steps,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            **common_kwargs
        )
    
    elif algorithm == 'DQN':
        return DQN(
            policy='MultiInputPolicy',
            batch_size=config.batch_size,
            gamma=config.gamma,
            learning_starts=1000,
            target_update_interval=1000,
            train_freq=4,
            **common_kwargs
        )
    
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def train_model(config: TrainingConfig, output_dir: str) -> Tuple[Any, Dict[str, float]]:
    """Train the RL model"""
    
    logger.info(f"Starting training with {config.algorithm}")
    logger.info(f"Total timesteps: {config.total_timesteps}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create environment
    env = create_environment(config, seed=42)
    logger.info(f"Created environment with {config.n_envs} processes")
    
    # Create model
    model = create_model(config.algorithm, env, config)
    logger.info(f"Created {config.algorithm} model")
    
    # Create callbacks
    callback = CustomCallback(config, output_dir)
    
    # Initialize wandb if enabled
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            config=vars(config),
            name=f"trackpilot_{config.algorithm}_{np.random.randint(1000)}"
        )
        logger.info("Initialized wandb logging")
    
    # Train model
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        logger.info("Training completed successfully")
        
        # Save final model
        final_model_path = os.path.join(output_dir, 'final_model')
        model.save(final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
        
        # Get training metrics
        training_metrics = callback.metrics.get_summary()
        
        return model, training_metrics
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        if config.use_wandb:
            wandb.finish()


def evaluate_model(model: Any, config: TrainingConfig, n_episodes: int = 100) -> Dict[str, float]:
    """Evaluate trained model performance"""
    
    logger.info(f"Evaluating model over {n_episodes} episodes")
    
    # Create evaluation environment
    eval_env = create_environment(config, seed=123)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode % 10 == 0:
            logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}")
    
    eval_metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'max_reward': np.max(episode_rewards),
        'min_reward': np.min(episode_rewards)
    }
    
    logger.info("Evaluation Results:")
    for key, value in eval_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return eval_metrics


def main():
    parser = argparse.ArgumentParser(description='Train TrackPilot RL Agent')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to training configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='Output directory for models and logs')
    parser.add_argument('--algorithm', type=str, choices=['PPO', 'A2C', 'DQN'],
                      help='RL algorithm to use (overrides config)')
    parser.add_argument('--timesteps', type=int,
                      help='Total training timesteps (overrides config)')
    parser.add_argument('--no-eval', action='store_true',
                      help='Skip evaluation after training')
    parser.add_argument('--no-export', action='store_true',
                      help='Skip TorchScript export')
    parser.add_argument('--wandb', action='store_true',
                      help='Enable wandb logging')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load configuration
    with open(args.config, 'r') as f:
        if args.config.endswith('.yaml') or args.config.endswith('.yml'):
            config_dict = yaml.safe_load(f)
        else:
            config_dict = json.load(f)
    
    # Create training config
    config = TrainingConfig(config_dict)
    
    # Override with command line arguments
    if args.algorithm:
        config.algorithm = args.algorithm
    if args.timesteps:
        config.total_timesteps = args.timesteps
    if args.wandb:
        config.use_wandb = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logger.add(
        os.path.join(args.output_dir, "training.log"),
        rotation="10 MB",
        retention="7 days",
        level="INFO"
    )
    
    try:
        # Train model
        model, training_metrics = train_model(config, args.output_dir)
        
        # Evaluate model
        if not args.no_eval:
            eval_metrics = evaluate_model(model, config)
            
            # Save evaluation results
            with open(os.path.join(args.output_dir, 'eval_metrics.json'), 'w') as f:
                json.dump(eval_metrics, f, indent=2)
        
        # Export to TorchScript
        if not args.no_export:
            logger.info("Exporting model to TorchScript")
            export_path = os.path.join(args.output_dir, 'model.pt')
            export_to_torchscript(model, export_path)
            logger.info(f"Exported model to {export_path}")
        
        # Save training configuration
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(config), f, indent=2)
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()