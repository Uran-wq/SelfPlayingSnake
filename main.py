import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from environment import create_env
from impl1_hamiltonian import create_agent as create_hamiltonian_agent
from impl2_deviation import create_deviation_agent
from impl3_neural import train_dqn, evaluate_agent, get_default_config, create_agent as create_dqn_agent
from utils import run_episodes, print_results, compare_agents, plot_comparison


def run_impl1(grid_size: int = 20, n_episodes: int = 10, render: bool = False) -> dict:
    print("\n" + "=" * 60)
    print("RUNNING IMPLEMENTATION 1: Hamiltonian Cycle Follower")
    print("=" * 60)

    env = create_env(grid_size, render_mode="human" if render else "rgb_array")
    agent = create_hamiltonian_agent(grid_size)

    results = run_episodes(env, agent, n_episodes, model_type="rule")
    print_results("Impl1 (Hamiltonian)", results)

    return results


def run_impl2(grid_size: int = 20, n_episodes: int = 10, render: bool = False,
              epsilon: float = 0.1) -> dict:
    print("\n" + "=" * 60)
    print("RUNNING IMPLEMENTATION 2: Hamiltonian + Deviations")
    print("=" * 60)

    env = create_env(grid_size, render_mode="human" if render else "rgb_array")
    agent = create_deviation_agent(grid_size, epsilon=epsilon)

    results = run_episodes(env, agent, n_episodes, model_type="rule")
    print_results("Impl2 (Deviation)", results)

    return results


def run_impl3(grid_size: int = 20, n_episodes: int = 100, train: bool = True,
              render: bool = False, model_path: str = "models/dqn_snake.pt") -> dict:
    print("\n" + "=" * 60)
    print("RUNNING IMPLEMENTATION 3: Neural Network (DQN)")
    print("=" * 60)

    config = get_default_config()
    config.grid_size = grid_size
    config.save_path = model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if train:
        print("Training DQN agent...")
        agent = train_dqn(config, render=render)
    else:
        print(f"Loading model from {model_path}...")
        agent = create_dqn_agent(config)
        agent.load(model_path)
        agent.set_eval_mode()

    print(f"\nEvaluating DQN agent for {n_episodes} episodes...")
    results = evaluate_agent(agent, config, n_episodes=n_episodes)
    print_results("Impl3 (DQN)", results)

    return results


def visualize_impl(impl: int, grid_size: int, episodes: int, model_path: str, epsilon: float, fps: int):
    from utils.visualizer import visualize_episode
    from environment import create_env

    neural = (impl == 3)

    if impl == 1:
        env = create_env(grid_size)
        agent = create_hamiltonian_agent(grid_size)
        title = "Impl1: Hamiltonian Cycle"
    elif impl == 2:
        env = create_env(grid_size)
        agent = create_deviation_agent(grid_size, epsilon=epsilon)
        title = "Impl2: Hamiltonian + Deviations"
    elif impl == 3:
        config = get_default_config()
        config.grid_size = grid_size
        config.save_path = model_path
        env = create_env(grid_size)
        agent = create_dqn_agent(config)
        if os.path.exists(model_path):
            agent.load(model_path)
            title = "Impl3: DQN (trained)"
        else:
            print(f"Model not found at {model_path}. Train first with --impl 3 --train")
            return
        agent.set_eval_mode()
    else:
        env = create_env(grid_size)
        agent = create_hamiltonian_agent(grid_size)
        title = "Impl1: Hamiltonian Cycle"

    print(f"Visualizing {title} ({episodes} episodes)")
    print("Controls: [SPACE] Pause, [UP/DOWN] Speed, [ESC] Quit")
    results = visualize_episode(env, agent, title=title, fps=fps, n_episodes=episodes, neural=neural)

    print("\nResults:")
    for r in results:
        print(f"  Episode {r['episode']}: Score={r['score']}, Steps={r['steps']}")


def run_all(grid_size: int = 20, n_episodes: int = 100, train_dqn: bool = True,
            render: bool = False, dqn_model_path: str = "models/dqn_snake.pt") -> dict:
    results = {}

    results["Impl1 (Hamiltonian)"] = run_impl1(grid_size, min(n_episodes, 10), render)
    results["Impl2 (Deviation)"] = run_impl2(grid_size, min(n_episodes, 10), render)
    results["Impl3 (DQN)"] = run_impl3(grid_size, n_episodes, train_dqn, render, dqn_model_path)

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    compare_agents(results)

    try:
        fig = plot_comparison(results, "Snake AI Agent Comparison")
        fig.savefig("comparison.png", dpi=150, bbox_inches='tight')
        print("\nComparison plot saved to comparison.png")
    except Exception as e:
        print(f"Could not plot comparison: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Snake AI - Three Implementations")
    parser.add_argument("--impl", type=int, choices=[1, 2, 3, 0], default=0,
                        help="Which implementation to run (0 = all)")
    parser.add_argument("--grid-size", type=int, default=20, help="Grid size (default: 20)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes (default: 100)")
    parser.add_argument("--train", action="store_true", help="Train DQN (impl 3)")
    parser.add_argument("--no-train", dest="train", action="store_false", help="Don't train DQN")
    parser.add_argument("--render", action="store_true", help="Render games visually")
    parser.add_argument("--model-path", type=str, default="models/dqn_snake.pt",
                        help="Path to save/load DQN model")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for impl2 deviations")
    parser.add_argument("--visual", action="store_true", help="Real-time pygame visualization")
    parser.add_argument("--visual-episodes", type=int, default=3, help="Episodes to visualize (default: 3)")
    parser.add_argument("--fps", type=int, default=10, help="FPS for visualization (default: 10)")

    parser.set_defaults(train=True)

    args = parser.parse_args()

    if args.visual:
        visualize_impl(args.impl if args.impl != 0 else 1, args.grid_size,
                       args.visual_episodes, args.model_path, args.epsilon, args.fps)
    elif args.impl == 1:
        run_impl1(args.grid_size, args.episodes, args.render)
    elif args.impl == 2:
        run_impl2(args.grid_size, args.episodes, args.render, args.epsilon)
    elif args.impl == 3:
        run_impl3(args.grid_size, args.episodes, args.train, args.render, args.model_path)
    else:
        run_all(args.grid_size, args.episodes, args.train, args.render, args.model_path)


if __name__ == "__main__":
    main()
