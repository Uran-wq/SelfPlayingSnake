import numpy as np
from typing import List, Dict, Any
from environment import SnakeEnv


def run_episodes(env: SnakeEnv, agent, n_episodes: int = 100, model_type: str = "fc") -> Dict[str, Any]:
    scores = []
    lengths = []
    rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        if model_type == "fc" and hasattr(env, 'get_simplified_observation'):
            state = env.get_simplified_observation()
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            if model_type == "fc":
                action = agent.act(state, training=False)
            else:
                action = agent.act(env)
            next_state, reward, done, info = env.step(action)
            
            if model_type == "fc" and hasattr(env, 'get_simplified_observation'):
                next_state = env.get_simplified_observation()
            
            state = next_state
            total_reward += reward
            steps += 1
        
        scores.append(info["score"])
        lengths.append(steps)
        rewards.append(total_reward)
    
    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "max_score": int(np.max(scores)),
        "min_score": int(np.min(scores)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "mean_reward": float(np.mean(rewards)),
        "scores": scores,
        "lengths": lengths,
        "rewards": rewards,
    }


def print_results(name: str, results: Dict[str, Any]) -> None:
    print(f"\n{name} Results:")
    print(f"  Mean Score: {results['mean_score']:.2f} ± {results['std_score']:.2f}")
    print(f"  Max Score:  {results['max_score']}")
    print(f"  Min Score:  {results['min_score']}")
    print(f"  Mean Len:   {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"  Mean Reward: {results['mean_reward']:.2f}")


def compare_agents(results_dict: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("AGENT COMPARISON")
    print("=" * 60)
    print(f"{'Agent':<20} {'Mean Score':>12} {'Max Score':>10} {'Mean Len':>10}")
    print("-" * 60)
    
    for name, results in results_dict.items():
        print(f"{name:<20} {results['mean_score']:>12.2f} {results['max_score']:>10} {results['mean_length']:>10.1f}")
    
    print("=" * 60)