# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from texasholdem.game.action_type import ActionType
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F

class DuelingDQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        
        # Calculate value and advantages
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using the Dueling DQN formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

class RaiseAmountNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-4, model_dir=None, metrics_dir=None, plots_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Dueling DQN networks
        self.q_network = DuelingDQNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DuelingDQNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.action_dim = action_dim
        
        # Exploration parameters
        self.epsilon = 1.0
        # self.epsilon = 0.0
        self.epsilon_min = 0.01
        # self.epsilon_min = 0.0
        self.epsilon_decay = 0.995
        
        # Training metrics
        self.episode_rewards = []
        self.win_history = []
        self.losses = []
        
        # Save directories
        self.model_dir = model_dir
        self.metrics_dir = metrics_dir
        self.plots_dir = plots_dir
        
        # Add raise amount network
        self.raise_network = RaiseAmountNetwork(state_dim).to(self.device)
        self.raise_optimizer = optim.Adam(self.raise_network.parameters(), lr=learning_rate)
        self.raise_losses = []
        
    def select_action(self, state, valid_actions, stack_info=None):
        # Validate inputs
        if not valid_actions or all(not v for v in valid_actions.values()):
            raise ValueError("No valid actions available")
        
        # Create mapping from index to ActionType
        action_map = list(valid_actions.keys())
        
        # Get valid action indices
        valid_indices = [i for i, (_, valid) in enumerate(valid_actions.items()) if valid]
        
        if not valid_indices:
            raise ValueError("No valid actions available")
        
        # Epsilon-greedy selection
        if np.random.random() < self.epsilon:
            action_idx = np.random.choice(valid_indices)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            mask = torch.tensor([float('-inf') if i not in valid_indices else 0 
                               for i in range(self.action_dim)]).to(self.device)
            masked_q_values = q_values + mask
            
            action_idx = masked_q_values.argmax(1).item()
            if action_idx not in valid_indices:
                action_idx = np.random.choice(valid_indices)
        
        action_type = action_map[action_idx]
        
        # If it's a raise action, determine the raise amount
        if action_type == ActionType.RAISE:
            if stack_info is None:
                raise ValueError("stack_info must be provided for RAISE actions")
            
            # Get current bet and chips to call
            current_bet = int(stack_info['current_bet'])
            chips_to_call = int(stack_info['chips_to_call'])
            min_raise = int(stack_info['min_raise'])
            max_raise = int(stack_info['max_raise'])
            
            # Get percentage from network (0 to 1)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                raise_percentage = self.raise_network(state_tensor).squeeze().item()
                
                # Calculate raise amount based on percentage of valid range
                raise_range = max_raise - min_raise
                raise_amount = min_raise + (raise_range * raise_percentage)
                
                # Ensure it's within bounds
                raise_amount = min(max_raise, max(min_raise, raise_amount))
                
                # Convert to total amount (what we need to raise TO)
                total_amount = current_bet + chips_to_call + raise_amount
                
                # Final safety check
                total_amount = min(total_amount, max_raise)
            
            # Ensure final amount is an integer
            total_amount = int(total_amount)
            
            return (action_type, total_amount)
        
        return (action_type, None)
        
    def update(self, replay_buffer, batch_size, gamma=0.99):
        if len(replay_buffer) < batch_size:
            return 0.0
            
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Ensure action indices are within bounds
        action_indices = torch.LongTensor([
            min(action.value, self.action_dim - 1) for action in actions
        ]).to(self.device)
        
        states = torch.FloatTensor(states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Use action_indices instead of actions
        current_q = self.q_network(states).gather(1, action_indices.unsqueeze(1))
        
        # Double DQN: Use online network to select actions and target network to evaluate them
        with torch.no_grad():
            # Select actions using online network
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            # Evaluate actions using target network
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + gamma * next_q * (1 - dones.unsqueeze(1))
            
        # Compute loss and update
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
        
    def update_target_network(self):
        """Soft update of target network"""
        tau = 0.01  # Soft update parameter
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        
    def log_episode(self, episode_reward, won):
        self.episode_rewards.append(episode_reward)
        self.win_history.append(int(won))
        
    def save_metrics(self):
        """Save metrics to CSV files and generate plots"""
        if not self.metrics_dir or not self.plots_dir:
            return
        
        # Save metrics to CSV
        metrics = {
            'rewards.csv': self.episode_rewards,
            'wins.csv': self.win_history,
            'losses.csv': self.losses
        }
        
        for filename, data in metrics.items():
            filepath = os.path.join(self.metrics_dir, filename)
            np.savetxt(filepath, data, delimiter=',')
        
        # Generate and save plots
        self.plot_metrics()
        
    def plot_metrics(self):
        """Plot metrics and save to files"""
        if not self.plots_dir:
            return
        
        def exponential_moving_average(data, alpha=0.1):
            ema = []
            if len(data) > 0:
                ema.append(data[0])
                for value in data[1:]:
                    ema.append(alpha * value + (1 - alpha) * ema[-1])
            return ema

        # Plot rewards
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, alpha=0.3, color='lightblue', label='Raw Rewards')
        smoothed_rewards = exponential_moving_average(self.episode_rewards)
        plt.plot(smoothed_rewards, color='blue', label='Smoothed Rewards')
        plt.title("Learning Curve")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'learning_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot win rate
        plt.figure(figsize=(10, 5))
        window = 100
        win_rate = np.convolve(self.win_history, np.ones(window)/window, mode='valid')
        plt.plot(win_rate, color='green', label=f'{window}-Episode Win Rate')
        plt.title("Win Rate")
        plt.xlabel("Episode")
        plt.ylabel("Win Rate")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'win_rate.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot losses
        if self.losses:
            plt.figure(figsize=(10, 5))
            plt.plot(self.losses, alpha=0.3, color='salmon', label='Raw Loss')
            smoothed_losses = exponential_moving_average(self.losses)
            plt.plot(smoothed_losses, color='red', label='Smoothed Loss')
            plt.title("Training Loss")
            plt.xlabel("Update Step")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.plots_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
    def save_model(self, filename):
        """Save model checkpoint"""
        if not self.model_dir:
            return
        
        filepath = os.path.join(self.model_dir, f"{filename}.pth")
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'win_history': self.win_history,
            'losses': self.losses
        }, filepath)

    def load_model(self, filename):
        """Load model checkpoint"""
        if not self.model_dir:
            return
        
        filepath = os.path.join(self.model_dir, f"{filename}.pth")
        checkpoint = torch.load(filepath)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.win_history = checkpoint['win_history']
        self.losses = checkpoint['losses']

    def get_raise_amount(self, state, player_stack, min_raise, max_raise):
        """
        Args:
            state: Current game state
            player_stack: Current player's stack size
            min_raise: Minimum raise amount
            max_raise: Maximum raise amount
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raise_percentage = self.raise_network(state_tensor).squeeze().item()
        
        # Calculate raw raise amount based on percentage of stack
        raw_raise = int(raise_percentage * player_stack)
        
        # Ensure raise amount is within valid range
        raise_amount = int(max(min_raise, min(raw_raise, max_raise)))
        return raise_amount
        
    def update_raise_network(self, states, raises, rewards, min_raises, max_raises):
        states = torch.FloatTensor(states).to(self.device)
        raises = torch.FloatTensor(raises).to(self.device)
        min_raises = torch.FloatTensor(min_raises).to(self.device)
        max_raises = torch.FloatTensor(max_raises).to(self.device)
        
        # Convert raise amounts to percentages based on valid ranges
        raise_ranges = max_raises - min_raises
        
        # Convert actual raises to percentages
        target_percentages = ((raises - min_raises) / raise_ranges).clamp(0, 1)
        target_percentages = target_percentages.unsqueeze(1)
        
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        # Get predicted percentages
        predicted_percentages = self.raise_network(states)
        
        # Calculate loss (MSE weighted by rewards)
        losses = F.mse_loss(predicted_percentages, target_percentages, reduction='none')
        weighted_loss = (losses * rewards.unsqueeze(1)).mean()
        
        # Update network
        self.raise_optimizer.zero_grad()
        weighted_loss.backward()
        self.raise_optimizer.step()
        
        self.raise_losses.append(weighted_loss.item())
        return weighted_loss.item()
