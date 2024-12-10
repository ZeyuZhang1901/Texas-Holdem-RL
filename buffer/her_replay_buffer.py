from typing import Dict, Tuple
import numpy as np
import torch
from texasholdem.game.action_type import ActionType
from collections import namedtuple
# Define experience tuple structure
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done', 'goal', 'achieved_goal'])

class HERBuffer:
    """Hindsight Experience Replay Buffer"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, 
            state: np.ndarray,
            action: ActionType,
            reward: float,
            next_state: np.ndarray,
            done: bool,
            goal: Dict,
            achieved_goal: Dict):
        """Store a transition"""
        # Input validation
        if not isinstance(state, np.ndarray) or not isinstance(next_state, np.ndarray):
            raise ValueError("State and next_state must be numpy arrays")
        if not isinstance(action, ActionType):
            raise ValueError("Action must be an ActionType")
        if not isinstance(goal, dict) or not isinstance(achieved_goal, dict):
            raise ValueError("Goals must be dictionaries")
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Experience(state, action, reward, next_state, 
                                              done, goal, achieved_goal)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences"""
        if batch_size > len(self.buffer):
            raise ValueError(f"Cannot sample {batch_size} experiences from buffer of size {len(self.buffer)}")
        
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        experiences = [self.buffer[i] for i in batch]
        
        try:
            states = torch.FloatTensor([e.state for e in experiences])
            actions = torch.LongTensor([e.action.value for e in experiences])
            rewards = torch.FloatTensor([e.reward for e in experiences])
            next_states = torch.FloatTensor([e.next_state for e in experiences])
            dones = torch.FloatTensor([e.done for e in experiences])
            goals = [e.goal for e in experiences]
            achieved_goals = [e.achieved_goal for e in experiences]
        except Exception as e:
            raise RuntimeError(f"Error converting experiences to tensors: {str(e)}")
        
        return states, actions, rewards, next_states, dones, goals, achieved_goals
        
    def __len__(self):
        return len(self.buffer)