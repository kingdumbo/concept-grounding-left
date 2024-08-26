import copy
import enum
import heapq
import numpy as np

import mini_behavior
from mini_behavior.utils.utils import AttrDict
from typing import Tuple, Dict, Any, Sequence


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left


def heuristic(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
    # Manhattan distance on a square grid
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def is_within_bounds(grid: np.ndarray, x: int, y: int) -> bool:
    return 0 <= x < len(grid) and 0 <= y < len(grid[0])


def face_goal(goal: Tuple[int, int, int],
              closest: Tuple[int, int, int]) -> Tuple[int, int, int]:
    # Determine the direction to face the original goal
    cx, cy, _ = closest
    gx, gy, _ = goal
    dx, dy = gx - cx, gy - cy
    if abs(dx) > abs(dy):
        direction = 2 if dx > 0 else 0  # face down if goal is below else up
    else:
        direction = 1 if dy > 0 else 3  # face right if goal is right else left
    return (cx, cy, direction)


def find_cycle(start, data):
    visited = set()
    current = start
    while current is not None:
        if current in visited:
            return True, current  # Cycle found
        visited.add(current)
        current = data.get(current)
    return False, None  # No cycle found


def reconstruct_path(
        came_from: Dict[Tuple[int, int, int], Any],
        current: Tuple[int, int, int]) -> Sequence[Tuple[int, int, int]]:
    path = []
    is_cycle, node_at = find_cycle(current, came_from)
    if is_cycle:
        raise ValueError(f"The A* Path contains a Cycle @ {node_at}")
    while current in came_from:
        path.append(current)
        current = copy.copy(came_from[current])
    path.reverse()
    return path


def a_star_search_modified(
        grid: np.ndarray,
        start: Tuple[int, int, int],
        goal: Tuple[int, int, int]) -> Sequence[Tuple[int, int, int]]:
    """
    Modified A* that can select the closest reachable point to the goal.

    For reachable goals this planner reaches it normally as A* would.

    Args:
        grid: 2D list of obstacles (1) and free spaces (0).
        start: Tuple (x, y, direction) as start position and orientation.
        goal: Tuple (x, y, direction) as the goal position and orientation.

    Returns:
        List of tuples as path from start to the modified goal.
    """

    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, goal), start))

    came_from = {start: None}
    cost_so_far = {start: 0}
    closest = start
    closest_dist = heuristic(start, goal)

    while open_list:
        _, current = heapq.heappop(open_list)

        # Update closest point (if closer to the goal)
        current_dist = heuristic(current, goal)
        if current_dist < closest_dist:
            closest = current
            closest_dist = current_dist

        if current[:2] == goal[:2]:  # check only x, y match
            goal = current   # update the goal to the reachable state
            break

        current_x, current_y, current_dir = current
        for i, (dx, dy) in enumerate(DIRS):
            nx, ny = current_x + dx, current_y + dy
            if is_within_bounds(grid, nx, ny) and grid[nx][ny] == 0:
                next_node = (nx, ny, i)
                new_cost = cost_so_far[current] + 1

                if (next_node not in (cost_so_far or (
                        new_cost < cost_so_far[next_node]))):
                    cost_so_far[next_node] = new_cost
                    heapq.heappush(open_list,
                                   (new_cost + heuristic(next_node, goal),
                                    next_node))
                    came_from[next_node] = current

    # If goal isn't directly reachable, use the closest reachable point
    if closest[:2] != goal[:2]:
        new_goal = face_goal(goal, closest)
        if new_goal != closest:
            came_from[new_goal] = closest
        return reconstruct_path(came_from, new_goal)

    print("Reachable goal")
    return reconstruct_path(came_from, goal)


def print_grid_with_path_and_direction(
        grid: np.ndarray,
        start: Tuple[int, int, int],
        goal: Tuple[int, int, int],
        path: Sequence[Tuple[int, int, int]]) -> None:
    # Define direction symbols
    direction_symbols = ['^', '>', 'v', '<']

    # Create a deep copy of the grid for visualization purposes
    visual_grid = [['#' if cell == 1 else '.' for cell in row] for row in grid]

    # Mark the path in the visual grid with directions
    for x, y, d in path:
        visual_grid[x][y] = direction_symbols[d]

    # Ensure start and goal symbols
    sx, sy, sd = start
    gx, gy, gd = goal
    visual_grid[sx][sy] = 'S'
    visual_grid[gx][gy] = 'G'

    # Print the visual grid
    print("Grid representation:")
    for row in visual_grid:
        print(' '.join(row))


def path_to_actions(path: Sequence[Tuple[int, int, int]],
                    actions: AttrDict) -> Sequence[enum.IntEnum]:
    """
    Converts a path consisting of (x, y, direction) into a sequence of actions.

    Actions are 'forward', 'left', 'right'.

    Args:
        path: List of tuples (x, y, direction)

    Returns:
        List of control actions
    """
    control_actions = []
    mapped_dict = {k: idx for idx, k in enumerate(DIRS)}
    for i in range(1, len(path)):
        prev_x, prev_y, prev_direction = path[i - 1]
        curr_x, curr_y, curr_direction = path[i]

        # Compute direction from previous node to current node
        dx, dy = curr_x - prev_x, curr_y - prev_y
        movement_direction = (dx, dy)

        mapped_direction = mapped_dict.get(movement_direction)

        # Determine the needed turn action
        additional = []
        if mapped_direction is not None:
            turn = (mapped_direction - prev_direction) % 4
            additional = [actions.forward]
        else:  # On Goal
            turn = (curr_direction - prev_direction) % 4

        if turn == 0:
            control_actions.extend(additional)
        elif turn == 1 or turn == -3:
            control_actions.append(actions.left)
            control_actions.extend(additional)
        elif turn == 2 or turn == -2:
            control_actions.extend(
                [actions.right, actions.right])
            control_actions.extend(additional)
        elif turn == 3 or turn == -1:
            control_actions.append(actions.right)
            control_actions.extend(additional)

    return control_actions
