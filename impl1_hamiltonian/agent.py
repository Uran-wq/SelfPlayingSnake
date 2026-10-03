from environment import SnakeEnv, Action
from .pathfinder import HamiltonianPathfinder, create_pathfinder


class HamiltonianAgent:
    def __init__(self, grid_size: int = 20):
        self.pathfinder = create_pathfinder(grid_size)
    
    def act(self, env: SnakeEnv) -> int:
        head_pos = env.snake[0]
        action = self.pathfinder.get_action_to_next(head_pos, env.direction)
        return int(action)
    
    def reset(self) -> None:
        pass


def create_agent(grid_size: int = 20) -> HamiltonianAgent:
    return HamiltonianAgent(grid_size)