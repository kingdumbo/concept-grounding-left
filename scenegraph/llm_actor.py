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
import pprint

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32
show_furniture = False
SEED = 1234


class LlmActor:
    def __init__(self, environment_string, time_per_step):
        # init env
        self.env = gym.make(environment_string)

        # init scenegraph
        self.sg = Scenegraph(self.env, self, time_per_step=time_per_step)
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
        #wandb.log({
        #    "command": command_string,
        #    "raw_parsing": raw_parsing,
        #    "output": output,
        #    "reward": reward,
        #    "task_done": done
        #})

        return output, reward, done

    def execute_task(self, task_obj, max_re_plans=1, max_steps = 5):
        # generate initial plan
        success_conds = task_obj.get("success_conditions", [])
        self.mission = task_obj.get("mission")
        assert self.mission is not None, "No task specified!"
        plan = self.llm.generate_plan(self.mission)
        print("PLAN:")
        for i, step in enumerate(plan):
            print(f"{i+1}. {step}")

        # execute until failure
        num_replans = 0
        num_steps = 0
        max_steps = len(success_conds) * 2 + 1
        done = False
        interrupted = False
        while num_steps < max_steps and not interrupted:
            for step in plan:
                self.current_step = step
                output, reward, done = self.execute_step(step)
                num_steps += 1
                if output != "Success":
                    interrupted = True
                    break
            else:
                done = True
                interrupted = True

        # evaluate
        num_passing_success_conds, num_success_conds, not_passinng_success_conds = self._evaluate_task(success_conds)
        result = {
            "mission": self.mission,
            "env": task_obj.get("env"),
            "all_steps": done,
            "num_planned_steps": len(plan),
            "num_steps_completed": num_steps,
            "max_steps": max_steps,
            "failed_step": self.current_step if not done else "",
            "success_conds": num_success_conds,
            "passing_conds": num_passing_success_conds,
            "success_rate": num_passing_success_conds/num_success_conds,
            "failed_conds": not_passinng_success_conds,
            "prompt_difficulty": task_obj.get("prompt_difficulty")
        }

        return result

    def _evaluate_task(self, success_conditions):
        num_success_conds = len(success_conditions)
        num_passing_success_conds = 0
        not_passing_success_conds = []

        for cond in success_conditions:
            cond["answer"] = "yes"
            output, _, _ = self._execute_raw_parsing(cond)
            if output == "yes":
                num_passing_success_conds += 1
            else:
                not_passing_success_conds.append(cond["question"])
        
        return num_passing_success_conds, num_success_conds, not_passing_success_conds

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
    with_wandb = False
    T = 0.1
    if with_wandb:
        wandb.init(
            project="Concept Grounding",
            
        )
    # load tasks
    tasks = parse_yaml("high_level_tasks.yaml")
    #tasks = [tasks[3]]

    # validate the success conditions
    env_string = "MiniGrid-CleaningUpTheKitchenOnly-16x16-N2-v0"
    llm_actor = LlmActor(env_string, T)
    with_error = []
    for task in tasks:
        for cond in task["success_conditions"]:
            cond["answer"] = "True"
            output, _, _ = llm_actor._execute_raw_parsing(cond)
            if output == "":
                with_error.append(cond)
    pprint.pp(with_error)
    if len(with_error) > 0:
        sys.exit()

    # execute tasks
    for task in tasks:
        print("TASK: " + task["mission"])
        env_string = task["env"]
        llm_actor = LlmActor(env_string, T)
        llm_actor.start_render()
        summary = llm_actor.execute_task(task)
        if with_wandb:
            wandb.log(summary)
        pprint.pp(summary)
