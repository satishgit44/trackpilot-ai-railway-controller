#!/usr/bin/env python3
"""
TrackPilot Imitation Learning Training Script

This script implements behavioral cloning and other imitation learning techniques
for railway traffic scheduling based on expert demonstrations and human overrides.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from loguru import logger
import wandb
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from models.networks import SchedulingActorCritic, ImitationNetwork
from models.railway_env import RailwaySchedulingEnv
from utils.data_processing import preprocess_demonstrations, augment_data
from utils.metrics import ImitationMetrics
from utils.export import export_to_torchscript


class ExpertDemonstrationDataset(Dataset):
    """Dataset for expert demonstrations and human override data"""
    
    def __init__(self, data_path: str, transform=None):
        """
        Initialize dataset from expert demonstration files
        
        Args:
            data_path: Path to demonstration data (JSON, CSV, or pickle)
            transform: Optional data transformation function
        """
        self.transform = transform
        self.data = self._load_data(data_path)
        self.states, self.actions, self.rewards = self._preprocess_data()
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load demonstration data from file"""
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                return json.load(f)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            return df.to_dict('records')
        elif data_path.endswith('.pkl') or data_path.endswith('.pickle'):
            with open(data_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    
    def _preprocess_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess demonstration data into tensors"""
        states = []
        actions = []
        rewards = []
        
        for demonstration in self.data:
            # Extract state features
            state_features = self._extract_state_features(demonstration)
            states.append(state_features)
            
            # Extract action (schedule decision)
            action = self._extract_action(demonstration)
            actions.append(action)
            
            # Extract reward/performance score
            reward = demonstration.get('performance_score', 0.0)
            rewards.append(reward)
        
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32)
        )
    
    def _extract_state_features(self, demonstration: Dict) -> List[float]:
        """Extract state features from demonstration"""
        features = []
        
        # Train features
        trains = demonstration.get('trains', [])
        for train in trains[:10]:  # Limit to first 10 trains
            features.extend([
                train.get('priority', 0),
                train.get('speed_kmh', 0),
                train.get('delay_minutes', 0),
                1 if train.get('is_freight', False) else 0
            ])
        
        # Pad if fewer than 10 trains
        while len(features) < 40:  # 10 trains * 4 features
            features.append(0.0)
        
        # Section features
        sections = demonstration.get('sections', [])
        for section in sections[:5]:  # Limit to first 5 sections
            features.extend([
                section.get('length_km', 0),
                section.get('max_trains', 0),
                section.get('current_occupancy', 0),
                1 if section.get('is_single_track', False) else 0
            ])
        
        # Pad if fewer than 5 sections
        while len(features) < 60:  # 40 train features + 5 sections * 4 features
            features.append(0.0)
        
        # Time features
        features.extend([
            demonstration.get('hour_of_day', 0) / 24.0,
            demonstration.get('day_of_week', 0) / 7.0,
            demonstration.get('weather_score', 0.5),  # 0-1 scale
            demonstration.get('traffic_density', 0.5)  # 0-1 scale
        ])
        
        return features
    
    def _extract_action(self, demonstration: Dict) -> int:
        """Extract action from demonstration"""
        # Convert scheduling decision to discrete action
        decision = demonstration.get('scheduling_decision', 'maintain')
        
        action_map = {
            'maintain': 0,
            'delay_train': 1,
            'priority_override': 2,
            'route_change': 3,
            'emergency_stop': 4
        }
        
        return action_map.get(decision, 0)
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        
        if self.transform:
            state = self.transform(state)
        
        return state, action, reward


class BehavioralCloningTrainer:
    """Behavioral cloning trainer for imitation learning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = ImitationNetwork(
            state_dim=config['model']['state_dim'],
            action_dim=config['model']['action_dim'],
            hidden_dims=config['model']['hidden_dims']
        ).to(self.device)
        
        # Initialize optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = ImitationMetrics()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['training']['lr_step_size'],
            gamma=config['training']['lr_gamma']
        )
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (states, actions, rewards) in enumerate(tqdm(dataloader, desc="Training")):
            states = states.to(self.device)
            actions = actions.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(states)
            loss = self.criterion(predictions, actions)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(predictions.data, 1)
            correct_predictions += (predicted == actions).sum().item()
            total_samples += actions.size(0)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct_predictions / total_samples
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for states, actions, rewards in tqdm(dataloader, desc="Evaluating"):
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                predictions = self.model(states)
                loss = self.criterion(predictions, actions)
                
                total_loss += loss.item()
                _, predicted = torch.max(predictions.data, 1)
                correct_predictions += (predicted == actions).sum().item()
                total_samples += actions.size(0)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct_predictions / total_samples
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, save_path: str) -> Dict[str, List[float]]:
        """Full training loop"""
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Record metrics
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(self.model.state_dict(), 
                          os.path.join(save_path, 'best_model.pth'))
                logger.info(f"New best model saved with accuracy: {best_val_acc:.4f}")
            
            # Wandb logging
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
        
        return history


class DAggerTrainer(BehavioralCloningTrainer):
    """Dataset Aggregation (DAgger) trainer for iterative imitation learning"""
    
    def __init__(self, config: Dict[str, Any], env: RailwaySchedulingEnv):
        super().__init__(config)
        self.env = env
        self.expert_policy = self._load_expert_policy()
    
    def _load_expert_policy(self):
        """Load or define expert policy for DAgger"""
        # This could be a rule-based expert or pre-trained model
        # For now, implement a simple heuristic expert
        def expert_policy(state):
            # Simple priority-based expert policy
            # In practice, this would be more sophisticated
            return np.random.choice(self.config['model']['action_dim'])
        
        return expert_policy
    
    def collect_trajectories(self, num_episodes: int) -> List[Tuple]:
        """Collect trajectories using current policy"""
        trajectories = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_data = []
            
            while True:
                # Get action from current policy
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    action_probs = self.model(state_tensor)
                    action = torch.argmax(action_probs).item()
                
                # Get expert action for this state
                expert_action = self.expert_policy(state)
                
                # Store state-action pair
                episode_data.append((state.copy(), expert_action))
                
                # Take action in environment
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                
                if done:
                    break
            
            trajectories.extend(episode_data)
        
        return trajectories
    
    def train_dagger(self, initial_data_path: str, num_iterations: int, 
                    episodes_per_iter: int, save_path: str) -> Dict[str, Any]:
        """Train using DAgger algorithm"""
        
        # Load initial demonstration data
        initial_dataset = ExpertDemonstrationDataset(initial_data_path)
        aggregated_data = [(s, a, r) for s, a, r in initial_dataset]
        
        logger.info(f"Starting DAgger training for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            logger.info(f"DAgger iteration {iteration + 1}")
            
            # Create dataset from aggregated data
            states = torch.stack([data[0] for data in aggregated_data])
            actions = torch.stack([data[1] for data in aggregated_data])
            rewards = torch.stack([data[2] for data in aggregated_data])
            
            # Split into train/val
            dataset_size = len(aggregated_data)
            train_size = int(0.8 * dataset_size)
            val_size = dataset_size - train_size
            
            train_dataset = torch.utils.data.TensorDataset(
                states[:train_size], actions[:train_size], rewards[:train_size]
            )
            val_dataset = torch.utils.data.TensorDataset(
                states[train_size:], actions[train_size:], rewards[train_size:]
            )
            
            train_loader = DataLoader(train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config['training']['batch_size'])
            
            # Train on current aggregated dataset
            history = self.train(train_loader, val_loader, 
                               self.config['training']['epochs_per_iteration'], save_path)
            
            # Collect new trajectories with current policy
            if iteration < num_iterations - 1:  # Don't collect on last iteration
                new_trajectories = self.collect_trajectories(episodes_per_iter)
                
                # Add to aggregated data
                for state, expert_action in new_trajectories:
                    state_tensor = torch.tensor(state, dtype=torch.float32)
                    action_tensor = torch.tensor(expert_action, dtype=torch.long)
                    reward_tensor = torch.tensor(0.0, dtype=torch.float32)  # Placeholder
                    aggregated_data.append((state_tensor, action_tensor, reward_tensor))
                
                logger.info(f"Added {len(new_trajectories)} new trajectories")
                logger.info(f"Total dataset size: {len(aggregated_data)}")
        
        return history


def plot_training_history(history: Dict[str, List[float]], save_path: str):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train TrackPilot Imitation Learning Model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to training configuration file')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to expert demonstration data')
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='Output directory for models and logs')
    parser.add_argument('--algorithm', type=str, choices=['bc', 'dagger'], default='bc',
                      help='Imitation learning algorithm')
    parser.add_argument('--epochs', type=int,
                      help='Number of training epochs (overrides config)')
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
        config = json.load(f)
    
    # Override config with arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.wandb:
        config['use_wandb'] = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logger.add(
        os.path.join(args.output_dir, "imitation_learning.log"),
        rotation="10 MB",
        retention="7 days",
        level="INFO"
    )
    
    # Initialize wandb if enabled
    if config.get('use_wandb', False):
        wandb.init(
            project=config.get('wandb_project', 'trackpilot-imitation'),
            config=config,
            name=f"imitation_{args.algorithm}_{np.random.randint(1000)}"
        )
    
    try:
        if args.algorithm == 'bc':
            # Behavioral Cloning
            logger.info("Starting Behavioral Cloning training")
            
            # Load dataset
            dataset = ExpertDemonstrationDataset(args.data)
            
            # Split dataset
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config['training']['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['training']['batch_size']
            )
            
            # Train model
            trainer = BehavioralCloningTrainer(config)
            history = trainer.train(
                train_loader, val_loader,
                config['training']['epochs'],
                args.output_dir
            )
            
        elif args.algorithm == 'dagger':
            # DAgger
            logger.info("Starting DAgger training")
            
            # Create environment
            env = RailwaySchedulingEnv()
            
            # Train with DAgger
            trainer = DAggerTrainer(config, env)
            history = trainer.train_dagger(
                args.data,
                config['dagger']['num_iterations'],
                config['dagger']['episodes_per_iteration'],
                args.output_dir
            )
        
        # Plot training history
        plot_training_history(history, args.output_dir)
        
        # Export model to TorchScript
        logger.info("Exporting model to TorchScript")
        model_path = os.path.join(args.output_dir, 'best_model.pth')
        
        if os.path.exists(model_path):
            # Load best model
            trainer.model.load_state_dict(torch.load(model_path))
            trainer.model.eval()
            
            # Export to TorchScript
            export_path = os.path.join(args.output_dir, 'model.pt')
            dummy_input = torch.randn(1, config['model']['state_dim']).to(trainer.device)
            traced_model = torch.jit.trace(trainer.model, dummy_input)
            traced_model.save(export_path)
            logger.info(f"Exported model to {export_path}")
        
        # Save final configuration
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Imitation learning training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        if config.get('use_wandb', False):
            wandb.finish()


if __name__ == '__main__':
    main()