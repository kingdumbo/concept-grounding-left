import torch
import numpy as np
import time
from mini_behavior.minibehavior import MiniBehaviorEnv
from mini_behavior.envs import CleaningUpTheKitchenOnlyEnv

from pathlib import Path
import sys

BASEPATH = Path(__file__).resolve().parent
sys.path.append(BASEPATH)
import grid_planner 

# time per step in s
T = 0.5

class Actor:
    def __init__(self, env, renderer):
        self.env = env
        self.unary_actions = {"pick": self._pick, "place": self._place}
        self.binary_actions = {}

        self.reward = None
        self.task_state = None
        
        self.actions = MiniBehaviorEnv.Actions

        self.renderer = renderer

        self.status_list = ["Success", "Failure", "Type of action cannot be performed", "Couldn't be reached", "Nothing to place"]

    def get_actions(self):
        return [key for key in {**self.unary_actions, **self.binary_actions}.keys()]

    def get_status_list(self):
        return self.status_list

    
    def act(self, action, object_1, object_2=None):
        if action in self.unary_actions:
            status, reward, done = self.unary_actions[action](object_1)
            return self.vectorize_output(status), reward, done
        elif action in self.binary_actions:
            assert object_2 is not None
            status, reward, done = self.binary_actions[action](object_1, object_2)
            return self.vectorize_output(status), reward, done
        else:
            raise NotImplementedError(f"Action: {action} is not implemented. Please choose one of {self.get_actions()}")

    def _pick(self, object_1):
        reward = 0
        done = False
        # assert that object can be picked up
        if "pickup" not in object_1.actions:
            return 2, reward, done
        # locate object in env
        target_position = object_1.cur_pos

        # move there
        reached_target = self.navigate_to(target_position, look_at=True)
        if not reached_target:
            return 3, reward, done # not reached

        # pick object
        _ = self.env.gen_obs()
        opened = False
        if object_1.inside_of is not None:
            self.env.step(self.actions.open)
            if self.renderer:
                self.renderer._redraw()
            time.sleep(T)
            opened = True

        assert self.env.mode == "primitive"

        # try different pick actions until object is carried
        picked = False
        pick_actions = [self.actions.pickup_0, self.actions.pickup_1, self.actions.pickup_2]
        for action in pick_actions:
            _, reward, done, _ = self.env.step(action)
            if object_1 in self.env.carrying:
                picked = True
                if self.renderer:
                    self.renderer._redraw()
                time.sleep(T)
                break

        if opened:
            self.env.step(action.close)
            if self.renderer:
                self.renderer._redraw()
            time.sleep(T)

        return int(not picked), reward, done

    def _place(self, object_1):
        reward = 0
        done = False
        # assert that there is something to place
        something_to_place = False
        for obj in self.env.carrying:
            if "drop" in obj.actions or "dropin" in obj.actions:
                something_to_place = True
                break
        if not something_to_place:
            return 4, reward, done # nothing to place

        # locate targe-object in env
        target_position = object_1.cur_pos

        # move there
        reached_target = self.navigate_to(target_position, look_at=True)
        if not reached_target:
            return 3, reward, done # not reached

        # place object
        _ = self.env.gen_obs()
        # target may havbe to be opened first
        opened = False
        breakpoint()
        if "open" in object_1.actions:
            self.env.step(self.actions.open)
            if self.renderer:
                self.renderer._redraw()
            time.sleep(T)
            opened = True

        assert self.env.mode == "primitive"
        placed = False
        carrying_before = [obj for obj in self.env.carrying]
        place_actions = [self.actions.drop_0, self.actions.drop_1, self.actions.drop_2, self.actions.drop_in]
        for action in place_actions:
            _, reward, done, _ = self.env.step(action)
            for obj in carrying_before:
                if obj.inside_of:
                    placed = True
                    if self.renderer:
                        self.renderer._redraw()
                    time.sleep(T)

        if opened:
            self.env.step(self.actions.close)
            if self.renderer:
                self.renderer._redraw()
            time.sleep(T)
        
        return int(not placed), reward, done


    def vectorize_output(self, status):
        len_status = len(self.status_list)
        output = torch.zeros(len_status)
        output[status] = 1
        return output
    
    def navigate_to(self, target_position, look_at=True):
        attempted_pos_orient = (*target_position, 0)
        try:
            start = (*self.env.agent_pos, self.env.agent_dir)
            grid = self._get_grid()
            path = grid_planner.a_star_search_modified(grid, start, attempted_pos_orient)
            grid_planner.print_grid_with_path_and_direction(grid, start, attempted_pos_orient, path)
            path_actions = grid_planner.path_to_actions(path, self.actions)
            for action in path_actions:
                self.env.step(action)
                if self.renderer:
                    self.renderer._redraw()
                time.sleep(T)
            return True
        except IndexError as e:
            print(e)
            return False
        return False

    def _get_grid(self):
        behavior_grid = self.env.grid
        width = behavior_grid.width
        height = behavior_grid.height
        grid = np.ones((width, height))
        for i in range(width):
            for j in range(height):
                if behavior_grid.is_empty(i,j):
                    grid[i,j] = 0
        return grid

