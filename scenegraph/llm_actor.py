import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the necessary directories to sys.path
sys.path.append(os.path.join(current_dir, '..', 'Jacinle'))
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(current_dir)

from scenegraph import Scenegraph
from scenegraph_oracle import ScenegraphOracle
from utils import load_questionbank
import pandas as pd
from gym_minigrid.wrappers import *
from mini_behavior.envs import CleaningUpTheKitchenOnlyEnv
from mini_behavior.window import Window

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32
show_furniture = False
SEED = 1234


class LlmActor:
    def __init__(self, environment_string):
        # init env
        self.env = gym.make(environment_string)

        # init scenegraph
        self.sg = Scenegraph(self.env)
        self.sg.update()

        # get domain and init oracle
        self.domain = self.sg.get_domain()
        self.oracle_actor = ScenegraphOracle(self.domain, self.sg)

        # window for display
        self.window = Window(f"mini_behavior - LLM-Actor - {environment_string}")
        self.render = False

        # mission
        self.mission = self.env.mission
        self.reset()

    def reset(self):
        # reset env
        self.env.reset()

        # reset scene graph
        self.sg.update()
        
        # reset mission
        if hasattr(self.env, 'mission'):
            print('Mission: %s' % self.env.mission)
            self.window.set_caption(self.env.mission)

        # reset window
        if self.render:
            self._redraw()

    def start_render(self):
        self.render = True
        self.sg.render(continual_rendering=True)
        self._redraw()
        self.window.show(block=False)
        

    def execute(self, command):
        try:
            if not isinstance(command, list):
                command = [command]
            output = self.oracle_actor.tell(command)
            self.sg.update()
            if self.render:
                self.sg.render(continual_rendering=True)
                self._redraw()
            output = output["pred_answer"][0]
            return output
        except Exception as e:
            raise e
            

    def _redraw(self):
        img = self.env.render('rgb_array', tile_size=TILE_PIXELS)
        self.window.no_closeup()
        self.window.set_inventory(self.env)
        self.window.show_img(img)

if __name__ == "__main__":
    env_string = "MiniGrid-CleaningUpTheKitchenOnly-16x16-N2-v0"
    llm_actor = LlmActor(env_string)
    llm_actor.start_render()
    commands = [{
        "question": "Pick up the broom.",
        "answer": "Success",
        "raw_parsing": "execute(Action, lambda k: pick(k, iota(Object, lambda y: broom(y))))"
    }]
    for command in commands:
        input()
        print(llm_actor.execute(command))
    input()
