import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from environment import SnakeEnv


def render_game(env: SnakeEnv, title: str = "Snake Game") -> None:
    img = env.render()
    if img is not None:
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"{title} - Score: {env.score}")
        plt.axis('off')
        plt.show()


def plot_training_curves(rewards: List[float], scores: List[int], lengths: List[int], 
                          window: int = 100, title: str = "Training Progress") -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    def moving_avg(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    axes[0].plot(rewards, alpha=0.3, label='Raw')
    if len(rewards) >= window:
        axes[0].plot(moving_avg(rewards, window), label=f'MA({window})', color='red')
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(scores, alpha=0.3, label='Raw')
    if len(scores) >= window:
        axes[1].plot(moving_avg(scores, window), label=f'MA({window})', color='red')
    axes[1].set_title('Episode Scores')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(lengths, alpha=0.3, label='Raw')
    if len(lengths) >= window:
        axes[2].plot(moving_avg(lengths, window), label=f'MA({window})', color='red')
    axes[2].set_title('Episode Lengths')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Steps')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_comparison(results: Dict[str, Dict[str, Any]], title: str = "Agent Comparison") -> Any:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    names = list(results.keys())
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    mean_scores = [results[n].get('mean_score', 0) for n in names]
    std_scores = [results[n].get('std_score', 0) for n in names]
    max_scores = [results[n].get('max_score', 0) for n in names]
    mean_lengths = [results[n].get('mean_length', 0) for n in names]
    
    axes[0, 0].bar(names, mean_scores, yerr=std_scores, capsize=5, color=colors[:len(names)], alpha=0.7)
    axes[0, 0].set_title('Mean Score (±std)')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].bar(names, max_scores, color=colors[:len(names)], alpha=0.7)
    axes[0, 1].set_title('Max Score')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    axes[1, 0].bar(names, mean_lengths, color=colors[:len(names)], alpha=0.7)
    axes[1, 0].set_title('Mean Episode Length')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    for i, name in enumerate(names):
        if 'scores' in results[name]:
            axes[1, 1].hist(results[name]['scores'], bins=20, alpha=0.5, label=name, color=colors[i])
    axes[1, 1].set_title('Score Distribution')
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def save_plot(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches='tight')