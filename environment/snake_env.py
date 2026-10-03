import numpy as np
from typing import Tuple, List, Optional
from enum import IntEnum


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class SnakeEnv:
    def __init__(self, grid_size: int = 20, render_mode: str = "rgb_array"):
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        self.snake: List[Tuple[int, int]] = []
        self.food: Tuple[int, int] = (0, 0)
        self.direction = Action.RIGHT
        self.score = 0
        self.steps = 0
        self.done = False
        
        self._action_to_delta = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
        }
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        self.snake = [(self.grid_size // 2, self.grid_size // 2 - i) for i in range(3)]
        self.direction = Action.RIGHT
        self.score = 0
        self.steps = 0
        self.done = False
        self._spawn_food()
        return self._get_observation()
    
    def _spawn_food(self) -> None:
        empty_cells = [(r, c) for r in range(self.grid_size) 
                       for c in range(self.grid_size) 
                       if (r, c) not in self.snake]
        self.food = empty_cells[np.random.randint(len(empty_cells))]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            return self._get_observation(), 0.0, True, {"score": self.score}
        
        if not self._is_valid_action(action):
            action = int(self.direction)
        
        self.direction = Action(action)
        self.steps += 1
        
        head_r, head_c = self.snake[0]
        dr, dc = self._action_to_delta[self.direction]
        new_head = (head_r + dr, head_c + dc)
        
        reward = 0.0
        
        if self._is_collision(new_head):
            self.done = True
            reward = -10.0
            return self._get_observation(), reward, True, {"score": self.score}
        
        self.snake.insert(0, new_head)
        
        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self._spawn_food()
        else:
            self.snake.pop()
            reward = 0.1
        
        if self.steps > self.grid_size * self.grid_size * 2:
            self.done = True
        
        return self._get_observation(), reward, self.done, {"score": self.score}
    
    def _is_valid_action(self, action: int) -> bool:
        if action == Action.UP and self.direction == Action.DOWN:
            return False
        if action == Action.DOWN and self.direction == Action.UP:
            return False
        if action == Action.LEFT and self.direction == Action.RIGHT:
            return False
        if action == Action.RIGHT and self.direction == Action.LEFT:
            return False
        return True
    
    def _is_collision(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return True
        if pos in self.snake:
            return True
        return False
    
    def _get_observation(self) -> np.ndarray:
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        for i, (r, c) in enumerate(self.snake):
            if i == 0:
                obs[r, c, 0] = 1.0
            else:
                obs[r, c, 1] = 1.0
        
        obs[self.food[0], self.food[1], 2] = 1.0
        
        return obs
    
    def get_simplified_observation(self) -> np.ndarray:
        head_r, head_c = self.snake[0]
        food_r, food_c = self.food
        
        danger_ahead = self._check_danger_straight()
        danger_left = self._check_danger_left()
        danger_right = self._check_danger_right()
        
        food_dx = food_c - head_c
        food_dy = food_r - head_r
        
        dir_vec = self._action_to_delta[self.direction]
        head_dx, head_dy = dir_vec[1], dir_vec[0]
        
        dist_to_food = abs(food_dx) + abs(food_dy)
        
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[self.direction] = 1.0
        
        features = np.array([
            float(danger_ahead),
            float(danger_left),
            float(danger_right),
            float(food_dx),
            float(food_dy),
            float(head_dx),
            float(head_dy),
            float(dist_to_food),
        ], dtype=np.float32)
        
        return np.concatenate([features, dir_onehot])
    
    def _check_danger_straight(self) -> bool:
        head_r, head_c = self.snake[0]
        dr, dc = self._action_to_delta[self.direction]
        return self._is_collision((head_r + dr, head_c + dc))
    
    def _check_danger_left(self) -> bool:
        head_r, head_c = self.snake[0]
        # left_dir = (self.direction + 2) % 4
        left_dir: Action
        if self.direction == Action.UP:
            left_dir = Action.LEFT
        elif self.direction == Action.DOWN:
            left_dir = Action.RIGHT
        elif self.direction == Action.LEFT:
            left_dir = Action.DOWN
        elif self.direction == Action.RIGHT:
            left_dir = Action.UP
        else:
            left_dir = (self.direction + 2) % 4
        
        dr, dc = self._action_to_delta[left_dir]
        return self._is_collision((head_r + dr, head_c + dc))
    
    def _check_danger_right(self) -> bool:
        head_r, head_c = self.snake[0]
        # right_dir = (self.direction + 2) % 4
        right_dir: Action
        if self.direction == Action.UP:
            right_dir = Action.RIGHT
        elif self.direction == Action.DOWN:
            right_dir = Action.LEFT
        elif self.direction == Action.LEFT:
            right_dir = Action.UP
        elif self.direction == Action.RIGHT:
            right_dir = Action.DOWN
        else:
            right_dir = (self.direction + 3) % 4
        
        dr, dc = self._action_to_delta[right_dir]
        return self._is_collision((head_r + dr, head_c + dc))
    
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
            
            for i, (r, c) in enumerate(self.snake):
                if i == 0:
                    img[r, c] = [0, 255, 0]
                else:
                    img[r, c] = [0, 200, 0]
            
            img[self.food[0], self.food[1]] = [255, 0, 0]
            
            return img
        elif self.render_mode == "human":
            self._render_text()
        return None
    
    def _render_text(self) -> None:
        grid = np.full((self.grid_size, self.grid_size), " . ", dtype=object)
        
        for i, (r, c) in enumerate(self.snake):
            if i == 0:
                grid[r, c] = " H "
            else:
                grid[r, c] = " # "
        
        grid[self.food[0], self.food[1]] = " F "
        
        print(f"Score: {self.score} | Steps: {self.steps}")
        for row in grid:
            print("".join(row))
        print()
    
    def clone(self) -> "SnakeEnv":
        new_env = SnakeEnv(self.grid_size, self.render_mode)
        new_env.snake = self.snake.copy()
        new_env.food = self.food
        new_env.direction = self.direction
        new_env.score = self.score
        new_env.steps = self.steps
        new_env.done = self.done
        return new_env


def create_env(grid_size: int = 20, render_mode: str = "rgb_array") -> SnakeEnv:
    return SnakeEnv(grid_size, render_mode)