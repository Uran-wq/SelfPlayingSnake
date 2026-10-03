from typing import List, Tuple, Dict
import numpy as np
from environment import Action


def build_hamiltonian_cycle(grid_size: int) -> List[Tuple[int, int]]:
    if grid_size % 2 != 0:
        raise ValueError("Grid size must be even for Hamiltonian cycle")
    
    cycle = []
    
    for col in range(grid_size):
        cycle.append((0, col))
    
    for row in range(1, grid_size):
        if row % 2 == 1:
            for col in range(grid_size - 1, 0, -1):
                cycle.append((row, col))
        else:
            for col in range(1, grid_size):
                cycle.append((row, col))
    
    for row in range(grid_size - 1, 0, -1):
        cycle.append((row, 0))
    
    return cycle


def build_cycle_index_map(cycle: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    return {pos: idx for idx, pos in enumerate(cycle)}


class HamiltonianPathfinder:
    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.cycle = build_hamiltonian_cycle(grid_size)
        self.cycle_map = build_cycle_index_map(self.cycle)
        self.cycle_len = len(self.cycle)
    
    def get_next_on_cycle(self, current_pos: Tuple[int, int]) -> Tuple[int, int]:
        idx = self.cycle_map[current_pos]
        next_idx = (idx + 1) % self.cycle_len
        return self.cycle[next_idx]
    
    def get_prev_on_cycle(self, current_pos: Tuple[int, int]) -> Tuple[int, int]:
        idx = self.cycle_map[current_pos]
        prev_idx = (idx - 1) % self.cycle_len
        return self.cycle[prev_idx]
    
    def get_cycle_index(self, pos: Tuple[int, int]) -> int:
        return self.cycle_map[pos]
    
    def get_action_to_next(self, current_pos: Tuple[int, int], current_dir: Action) -> Action:
        next_pos = self.get_next_on_cycle(current_pos)
        return self._pos_to_action(current_pos, next_pos)
    
    def _pos_to_action(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> Action:
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        
        if dr == -1 and dc == 0:
            return Action.UP
        elif dr == 1 and dc == 0:
            return Action.DOWN
        elif dr == 0 and dc == -1:
            return Action.LEFT
        elif dr == 0 and dc == 1:
            return Action.RIGHT
        else:
            raise ValueError(f"Invalid transition: {from_pos} -> {to_pos}")
    
    def get_distance_on_cycle(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        from_idx = self.cycle_map[from_pos]
        to_idx = self.cycle_map[to_pos]
        
        if to_idx >= from_idx:
            return to_idx - from_idx
        else:
            return self.cycle_len - (from_idx - to_idx)
    
    def is_on_cycle(self, pos: Tuple[int, int]) -> bool:
        return pos in self.cycle_map


def create_pathfinder(grid_size: int = 20) -> HamiltonianPathfinder:
    return HamiltonianPathfinder(grid_size)