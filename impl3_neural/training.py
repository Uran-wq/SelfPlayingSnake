import os
import numpy as np
import torch
from environment import create_env
from .agent import DQNAgent, create_agent
from .config import DQNConfig, get_default_config


def train_dqn(config: DQNConfig | None = None, render: bool = False) -> DQNAgent:
    if config is None:
        config = get_default_config()
    
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
    
    env = create_env(config.grid_size, render_mode="rgb_array")
    agent = create_agent(config)
    
    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    
    print(f"Training on {config.device}")
    print(f"Model type: {config.model_type}")
    print(f"Episodes: {config.max_episodes}")
    print("-" * 50)
    
    for episode in range(config.max_episodes):
        state = env.reset()
        if config.model_type == "fc":
            state = env.get_simplified_observation()
        
        total_reward = 0
        steps = 0
        
        for step in range(config.max_steps_per_episode):
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            if config.model_type == "fc":
                next_state = env.get_simplified_observation()
            
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_scores.append(info["score"])
        episode_lengths.append(steps)
        
        if (episode + 1) % config.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-config.log_interval:])
            avg_score = np.mean(episode_scores[-config.log_interval:])
            avg_len = np.mean(episode_lengths[-config.log_interval:])
            print(f"Episode {episode + 1:5d} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Score: {avg_score:5.2f} | "
                  f"Avg Len: {avg_len:5.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        if (episode + 1) % 1000 == 0:
            agent.save(config.save_path.replace(".pt", f"_ep{episode+1}.pt"))
    
    agent.save(config.save_path)
    print(f"\nTraining complete. Model saved to {config.save_path}")
    
    return agent


def evaluate_agent(agent: DQNAgent, config: DQNConfig, n_episodes: int = 100) -> dict:
    env = create_env(config.grid_size, render_mode="rgb_array")
    agent.set_eval_mode()
    
    scores = []
    lengths = []
    
    for episode in range(n_episodes):
        state = env.reset()
        if config.model_type == "fc":
            state = env.get_simplified_observation()
        
        done = False
        steps = 0
        
        while not done:
            action = agent.act(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            if config.model_type == "fc":
                next_state = env.get_simplified_observation()
            
            state = next_state
            steps += 1
        
        scores.append(info["score"])
        lengths.append(steps)
    
    agent.set_train_mode()
    
    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "max_score": int(np.max(scores)),
        "min_score": int(np.min(scores)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "mean_reward": float(np.mean(scores)),
        "scores": scores,
        "lengths": lengths,
        "rewards": scores,
    }


if __name__ == "__main__":
    config = get_default_config()
    agent = train_dqn(config)
    results = evaluate_agent(agent, config, n_episodes=100)
    print(f"\nEvaluation Results:")
    print(f"  Mean Score: {results['mean_score']:.2f} ± {results['std_score']:.2f}")
    print(f"  Max Score: {results['max_score']}")
    print(f"  Mean Length: {results['mean_length']:.1f}")