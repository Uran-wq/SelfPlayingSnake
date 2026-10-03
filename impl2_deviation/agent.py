import random
import numpy as np
from environment import SnakeEnv, Action
from impl1_hamiltonian.pathfinder import HamiltonianPathfinder, create_pathfinder


class DeviationAgent:
    def __init__(
        self, 
        grid_size: int = 20,
        epsilon: float = 0.1,
        heuristic_threshold: int = 3
    ):
        self.pathfinder = create_pathfinder(grid_size)
        self.epsilon = epsilon
        self.heuristic_threshold = heuristic_threshold
        self.grid_size = grid_size
        
        self._action_to_delta = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
        }
    
    def act(self, env: SnakeEnv) -> int:
        head_pos = env.snake[0]
        food_pos = env.food
        
        if random.random() < self.epsilon:
            return self._random_safe_action(env)
        
        if self._should_deviate_to_food(head_pos, food_pos):
            return self._greedy_towards_food(env, head_pos, food_pos)
        
        action = self.pathfinder.get_action_to_next(head_pos, env.direction)
        return int(action)
    
    def _random_safe_action(self, env: SnakeEnv) -> int:
        head_pos = env.snake[0]
        safe_actions = []
        
        for action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
            if action == Action.UP and env.direction == Action.DOWN:
                continue
            if action == Action.DOWN and env.direction == Action.UP:
                continue
            if action == Action.LEFT and env.direction == Action.RIGHT:
                continue
            if action == Action.RIGHT and env.direction == Action.LEFT:
                continue
            
            dr, dc = self._action_to_delta[action]
            new_pos = (head_pos[0] + dr, head_pos[1] + dc)
            
            if not self._is_collision(env, new_pos):
                safe_actions.append(action)
        
        if safe_actions:
            return int(random.choice(safe_actions))
        
        return int(env.direction)
    
    def _should_deviate_to_food(self, head_pos: Tuple[int, int], food_pos: Tuple[int, int]) -> bool:
        manhattan_dist = abs(head_pos[0] - food_pos[0]) + abs(head_pos[1] - food_pos[1])
        cycle_dist = self.pathfinder.get_distance_on_cycle(head_pos, food_pos)
        
        return manhattan_dist < self.heuristic_threshold and manhattan_dist < cycle_dist
    
    def _greedy_towards_food(self, env: SnakeEnv, head_pos: Tuple[int, int], food_pos: Tuple[int, int]) -> int:
        best_action = int(env.direction)
        best_dist = float('inf')
        
        for action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
            if action == Action.UP and env.direction == Action.DOWN:
                continue
            if action == Action.DOWN and env.direction == Action.UP:
                continue
            if action == Action.LEFT and env.direction == Action.RIGHT:
                continue
            if action == Action.RIGHT and env.direction == Action.LEFT:
                continue
            
            dr, dc = self._action_to_delta[action]
            new_pos = (head_pos[0] + dr, head_pos[1] + dc)
            
            if self._is_collision(env, new_pos):
                continue
            
            dist = abs(new_pos[0] - food_pos[0]) + abs(new_pos[1] - food_pos[1])
            
            if dist < best_dist:
                best_dist = dist
                best_action = int(action)
        
        return best_action
    
    def _is_collision(self, env: SnakeEnv, pos: Tuple[int, int]) -> bool:
        r, c = pos
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return True
        if pos in env.snake:
            return True
        return False
    
    def reset(self) -> None:
        pass


def create_deviation_agent(
    grid_size: int = 20,
    epsilon: float = 0.1,
    heuristic_threshold: int = 3
) -> DeviationAgent:
    return DeviationAgent(grid_size, epsilon, heuristic_threshold)