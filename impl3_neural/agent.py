import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Tuple, List
from environment import SnakeEnv
from .model import DQN, create_model
from .config import DQNConfig


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.bool_)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.policy_net = create_model(config.model_type, 
                                       input_size=config.input_size,
                                       hidden_size=config.hidden_size,
                                       output_size=config.output_size,
                                       grid_size=config.grid_size).to(self.device)
        self.target_net = create_model(config.model_type,
                                       input_size=config.input_size,
                                       hidden_size=config.hidden_size,
                                       output_size=config.output_size,
                                       grid_size=config.grid_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        self.epsilon = config.epsilon_start
        self.steps_done = 0
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.output_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> float:
        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (self.config.gamma * next_q * ~dones)
        
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps_done += 1
        
        if self.steps_done % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        
        return loss.item()
    
    def save(self, path: str):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
    
    def set_eval_mode(self):
        self.policy_net.eval()
        self.epsilon = 0.0
    
    def set_train_mode(self):
        self.policy_net.train()


def create_agent(config: DQNConfig) -> DQNAgent:
    return DQNAgent(config)