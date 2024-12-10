# replay_buffer.py
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )
        
    def __len__(self):
        return len(self.buffer)
    
class RaiseReplayBuffer:
    def __init__(self, max_size=10000):
        self.states = []
        self.raise_amounts = []
        self.rewards = []
        self.min_raises = []
        self.max_raises = []
        self.max_size = max_size
        
    def push(self, state, raise_amount, reward, min_raise, max_raise):
        if len(self.states) >= self.max_size:
            self.states.pop(0)
            self.raise_amounts.pop(0)
            self.rewards.pop(0)
            self.min_raises.pop(0)
            self.max_raises.pop(0)
        
        self.states.append(state)
        self.raise_amounts.append(raise_amount)
        self.rewards.append(reward)
        self.min_raises.append(min_raise)
        self.max_raises.append(max_raise)
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.states), min(batch_size, len(self.states)), replace=False)
        return (
            [self.states[i] for i in indices],
            [self.raise_amounts[i] for i in indices],
            [self.rewards[i] for i in indices],
            [self.min_raises[i] for i in indices],
            [self.max_raises[i] for i in indices]
        )