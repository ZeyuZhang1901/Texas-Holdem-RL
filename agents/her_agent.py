import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from texasholdem.game.action_type import ActionType
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from collections import namedtuple
import torch.nn.functional as F

from buffer.her_replay_buffer import HERBuffer

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

class HERAgent:
    def __init__(self, 
                state_dim: int,
                action_dim: int,
                learning_rate: float = 1e-4,
                model_dir: Optional[str] = None,
                metrics_dir: Optional[str] = None,
                plots_dir: Optional[str] = None,
                k_goals: int = 4):
        """
        Initialize HER Agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            model_dir: Directory to save models
            metrics_dir: Directory to save metrics
            plots_dir: Directory to save plots
            k_goals: Number of additional goals to sample for each experience
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k_goals = k_goals
        
        # Initialize networks (reusing DuelingDQNetwork architecture)
        self.q_network = DuelingDQNetwork(state_dim + self.goal_dim(), action_dim).to(self.device)
        self.target_network = DuelingDQNetwork(state_dim + self.goal_dim(), action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.action_dim = action_dim
        
        # Exploration parameters
        self.epsilon = 0.1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Training metrics
        self.episode_rewards = []
        self.win_history = []
        self.losses = []
        
        # Save directories
        self.model_dir = model_dir
        self.metrics_dir = metrics_dir
        self.plots_dir = plots_dir
        
        # Initialize raise amount network
        self.raise_network = RaiseAmountNetwork(state_dim + self.goal_dim()).to(self.device)
        self.raise_optimizer = optim.Adam(self.raise_network.parameters(), lr=learning_rate)
        self.raise_losses = [] 

    def goal_dim(self) -> int:
        """Return dimension of goal space"""
        return 4  # [target_pot_size, target_fold_rate, target_hand_strength, target_stack]
    
    def compute_reward(self, 
                      achieved_goal: Dict,
                      desired_goal: Dict,
                      info: Dict) -> float:
        """
        Compute reward based on achieved vs desired goal
        
        Args:
            achieved_goal: Dictionary containing achieved metrics
            desired_goal: Dictionary containing target metrics
            info: Additional information about the state
        """
        reward = 0
        
        # Reward for achieving target pot size
        pot_diff = abs(achieved_goal['pot_size'] - desired_goal['pot_size'])
        reward += -0.5 * (pot_diff / info['big_blind'])
        
        # Reward for achieving target fold rate
        if achieved_goal['opponent_folded'] == desired_goal['opponent_folded']:
            reward += 1.0
            
        # Reward for hand strength preservation
        strength_diff = abs(achieved_goal['hand_strength'] - desired_goal['hand_strength'])
        reward += -0.5 * strength_diff
        
        # Reward for stack preservation
        stack_diff = abs(achieved_goal['stack_size'] - desired_goal['stack_size'])
        reward += -0.5 * (stack_diff / info['big_blind'])
        
        return reward
        
    def sample_goals(self, 
                    achieved_goals: List[Dict],
                    episode_transitions: List) -> List[Dict]:
        """
        Sample additional goals for HER
        
        Args:
            achieved_goals: List of achieved goals in episode
            episode_transitions: List of transitions in episode
        """
        goals = []
        
        # Strategy 1: Future strategy - sample goals that were achieved later in episode
        future_idx = np.random.randint(len(achieved_goals))
        goals.append(achieved_goals[future_idx])
        
        # Strategy 2: Final strategy - use the final outcome as goal
        goals.append(achieved_goals[-1])
        
        # Strategy 3: Random strategy - create synthetic goals
        for _ in range(self.k_goals - 2):
            random_goal = {
                'pot_size': np.random.uniform(0, max(g['pot_size'] for g in achieved_goals)),
                'opponent_folded': bool(np.random.randint(2)),
                'hand_strength': np.random.uniform(0, 1),
                'stack_size': np.random.uniform(
                    min(g['stack_size'] for g in achieved_goals),
                    max(g['stack_size'] for g in achieved_goals)
                )
            }
            goals.append(random_goal)
            
        return goals

    def select_action(self, state: np.ndarray, goal: Dict, valid_actions: Dict) -> Tuple[ActionType, Optional[float]]:
        """Select action using epsilon-greedy policy"""
        # Create mapping from index to ActionType
        action_map = list(valid_actions.keys())
        
        # Get valid action indices
        valid_indices = [i for i, (_, valid) in enumerate(valid_actions.items()) if valid]
        
        if not valid_indices:
            raise ValueError("No valid actions available")
        
        # Concatenate state and goal
        state_goal = self._concat_state_goal(state, goal)
        state_tensor = torch.FloatTensor(state_goal).unsqueeze(0).to(self.device)
        
        if np.random.random() < self.epsilon:
            action_idx = np.random.choice(valid_indices)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                
                # Mask invalid actions
                mask = torch.tensor([float('-inf') if i not in valid_indices else 0 
                                   for i in range(self.action_dim)]).to(self.device)
                masked_q_values = q_values + mask
                
                action_idx = masked_q_values.argmax(1).item()
                if action_idx not in valid_indices:
                    action_idx = np.random.choice(valid_indices)
        
        action_type = action_map[action_idx]
        
        if action_type == ActionType.RAISE:
            raise_amount = self.get_raise_amount(state_goal)
            return action_type, raise_amount
        return action_type, None

    def update(self, her_buffer: HERBuffer, batch_size: int) -> float:
        """Update networks using HER buffer"""
        if len(her_buffer) < batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones, goals, achieved_goals = her_buffer.sample(batch_size)
        
        # Convert to tensors and move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Concatenate states with goals
        state_goals = torch.cat([states, self._encode_goals(goals)], dim=1)
        next_state_goals = torch.cat([next_states, self._encode_goals(goals)], dim=1)
        
        # Get current Q values
        current_q_values = self.q_network(state_goals).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_goals).max(1)[0].unsqueeze(1)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * 0.99 * next_q_values
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def _concat_state_goal(self, state: np.ndarray, goal: Dict) -> np.ndarray:
        """Concatenate state and goal into single vector"""
        goal_vector = np.array([
            goal['pot_size'],
            float(goal['opponent_folded']),
            goal['hand_strength'],
            goal['stack_size']
        ])
        return np.concatenate([state, goal_vector])

    def _encode_goals(self, goals: List[Dict]) -> torch.Tensor:
        """Encode list of goal dictionaries into tensor"""
        goal_vectors = []
        for goal in goals:
            goal_vector = [
                goal['pot_size'],
                float(goal['opponent_folded']),
                goal['hand_strength'],
                goal['stack_size']
            ]
            goal_vectors.append(goal_vector)
        return torch.FloatTensor(goal_vectors).to(self.device)

    def update_target_network(self):
        """Update target network parameters"""
        self.target_network.load_state_dict(self.q_network.state_dict())