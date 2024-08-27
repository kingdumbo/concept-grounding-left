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
from gpt_prompting import PromptingOpenAI
from utils import load_questionbank
import pandas as pd
from gym_minigrid.wrappers import *
from mini_behavior.envs import CleaningUpTheKitchenOnlyEnv
from mini_behavior.window import Window
import yaml
import wandb

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32
show_furniture = False
SEED = 1234


class LlmActor:
    def __init__(self, environment_string):
        # init env
        self.env = gym.make(environment_string)

        # init scenegraph
        self.sg = Scenegraph(self.env, self)
        self.sg.update()

        # get domain and init oracle
        self.domain = self.sg.get_domain()
        self.oracle_actor = ScenegraphOracle(self.domain, self.sg)

        # window for display
        self.window = Window(f"mini_behavior - LLM-Actor - {environment_string}")
        self.render = False

        # mission
        self.mission = self.env.mission
        self.reward = 0
        self.task_done = False

        # the llm
        self.llm = PromptingOpenAI(
            high_level_prompt_filename="system_prompt_high_level.txt",
            initial_prompt_filename="system_prompt_actions_1.txt",
            second_prompt_template_filename="system_prompt_actions_2.txt",
            scenegraph=self.sg
        )

        self.reset()

    def reset(self):
        # reset env
        self.env.reset()

        # reset scene graph
        self.sg.update()
        
        # reset mission
        self.mission = ""
        self.current_step = ""

        # reset window
        if self.render:
            self._redraw()

    def start_render(self):
        self.render = True
        self.sg.render(continual_rendering=True)
        self._redraw()
        self.window.show(block=False)
        

    def _execute_raw_parsing(self, command):
        try:
            if not isinstance(command, list):
                command = [command]
            output = self.oracle_actor.tell(command)
            self.sg.update()
            if self.render:
                self.sg.render(continual_rendering=True)
                self._redraw()
            output = output["pred_answer"][0]
            self.reward, self.task_done = self.sg.get_reward_n_done()
            return output, self.reward, self.task_done
        except Exception as e:
            raise e
            
    def execute_step(self, command_string):
        question_dict = {
            "question": command_string,
            "answer": "Success"
        }
        # first get raw_parsing
        raw_parsing = self.llm.to_raw_parsing(command_string)
        question_dict["raw_parsing"] = raw_parsing

        # then execute
        output, reward, done = self._execute_raw_parsing(question_dict)
        print(f"{command_string} - {raw_parsing} - {output}")
        wandb.log({
            "command": command_string,
            "raw_parsing": raw_parsing,
            "output": output,
            "reward": reward,
            "task_done": done
        })

        return output, reward, done

    def execute_task(self, task_description, max_re_plans=1, max_steps = 5):
        # generate initial plan
        self.mission = task_description
        plan = self.llm.generate_plan(task_description)
        print("PLAN:")
        for i, step in enumerate(plan):
            print(f"{i+1}. {step}")

        # execute until failure
        num_replans = 0
        num_steps = 0
        done = False
        while num_replans < max_re_plans and num_steps < max_steps and not done:
            for step in plan:
                self.current_step = step
                output, reward, done = self.execute_step(step)
                num_steps += 1
                if output != "Success":
                    print("FAILURE: Step not sucessful, aborting task.")
                    wandb.log({"result": "FAILURE - Step not successful, aborting task."})
                    return None
                    plan = self.llm.update_plan(step, output) 
                    num_replans += 1
                    break
            done = True
        wandb.log({"result": "SUCCESS - All steps successfully executed."})


    def _redraw(self):
        img = self.env.render('rgb_array', tile_size=TILE_PIXELS)
        self.window.no_closeup()
        self.window.set_inventory(self.env)
        self.window.show_img(img)
        self.window.set_caption(f"STEP: {self.current_step}")
        self.window.fig.suptitle(f"TASK: {self.mission}")

def parse_yaml(filename):
    path = str(current_dir) + "/" + filename
    with open(path, "r") as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    wandb.init(
        project="Concept Grounding",
        
    )
    # load tasks
    tasks = parse_yaml("high_level_tasks.yaml")
    for task in tasks:
        print("TASK: " + task["mission"])
        env_string = task["env"]
        wandb.log({"env": env_string, "mission": task["mission"]})
        llm_actor = LlmActor(env_string)
        llm_actor.start_render()
        llm_actor.execute_task(task["mission"], max_steps = task["max_steps"])
        input()
