from dataclasses import dataclass


@dataclass
class DQNConfig:
    grid_size: int = 20
    
    model_type: str = "fc"
    input_size: int = 12
    hidden_size: int = 128
    output_size: int = 4
    
    learning_rate: float = 1e-3
    gamma: float = 0.99
    
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    
    buffer_size: int = 10000
    batch_size: int = 64
    target_update_freq: int = 1000
    
    max_episodes: int = 5000
    max_steps_per_episode: int = 2000
    
    save_path: str = "models/dqn_snake.pt"
    log_interval: int = 100
    
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"


def get_default_config() -> DQNConfig:
    return DQNConfig()