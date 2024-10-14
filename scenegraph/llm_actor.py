import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the necessary directories to sys.path
sys.path.append(os.path.join(current_dir, '..', 'Jacinle'))
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', "LEFT"))
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
import pprint
import matplotlib.pyplot as plt
import torch

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32
show_furniture = False

CONFIG = {
    "model":"gpt-4o-mini",
    "max_tokens": 256,
    "temperature": 0.2,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "response_format": {"type": "text"}
}


class LlmActor:
    def __init__(self, environment_string, time_per_step, seed=0, llm_config=CONFIG):
        # init env
        self.env = gym.make(environment_string)
        if seed:
            self.env.seed(seed)
            self.env.reset()

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
            scenegraph=self.sg,
            config=llm_config
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

    def stop_render(self):
        self.render = False
        self.window.closed = True
        plt.close(self.window.fig)
        

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
            if output == "":
                output = "Failed parsing"
            self.reward, self.task_done = self.sg.get_reward_n_done()
            return output, self.reward, self.task_done
        except Exception as e:
            raise e
            
    def execute_step(self, command_string, seed=None):
        question_dict = {
            "question": command_string,
            "answer": "Success"
        }
        # first get raw_parsing
        raw_parsing = self.llm.to_raw_parsing(command_string, seed=seed)
        question_dict["raw_parsing"] = raw_parsing

        # then execute
        output, reward, done = self._execute_raw_parsing(question_dict)
        print(f"{command_string} - {raw_parsing} - {output}")

        # for logging
        log = {
            "command": command_string,
            "raw_parsing": raw_parsing,
            "output": output,
            "reward": reward,
            "task_done": done
        }

        return output, reward, done, log

    def execute_task(self, task_obj, max_re_plans=1, max_steps = 5, step_logging_list=None):
        # extract info from task object
        seed = task_obj.get("seed")
        task_number = task_obj.get("task_number")
        difficulty = task_obj.get("prompt_difficulty")


        # generate initial plan
        success_conds = task_obj.get("success_conditions", [])
        self.mission = task_obj.get("mission")
        assert self.mission is not None, "No task specified!"
        plan = self.llm.generate_plan(self.mission, seed)
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
                output, reward, done, log = self.execute_step(step, seed)
                if step_logging_list is not None:
                    log["task_number"] = task_number
                    log["difficulty"] = difficulty
                    log["step_number"] = num_steps
                    step_logging_list.append(log)
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
            "title": task_obj.get("title"),
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
            "prompt_difficulty": difficulty,
            "final_output": output,
            "plan": plan
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
    T = 0
    LLM_CONFIG = {
        "model":"gpt-4o",
        "max_tokens": 256,
        "temperature": 0.2,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": {"type": "text"}
    }

    # load tasks
    tasks = parse_yaml("high_level_tasks.yaml")

    with_error = []
    prev_env_string = ""
    llm_actor = None
    for task in tasks:
        env_string = task["env"]
        if not env_string == prev_env_string:
            env_string = task["env"]
            llm_actor = LlmActor(env_string, T)
            prev_env_string = env_string
        for cond in task["success_conditions"]:
            cond["answer"] = "True"
            output, _, _ = llm_actor._execute_raw_parsing(cond)
            if output == "":
                with_error.append(cond)
        llm_actor.stop_render()
    pprint.pp(with_error)
    if len(with_error) > 0:
        print(with_error)
        sys.exit()

    # re-run with different seeds
    seeds = [123, 456, 789, 101112, 131415]

    # constrain for test
    #seeds = seeds[:1]
    #seeds = [seeds[0]]
    #tasks = tasks[4:5]

    # execute tasks
    data = []
    step_logging = []

    for seed in seeds:
        torch.manual_seed(seed)
        for i, task in enumerate(tasks):
            for task_formulation in task["prompts"]:
                try:
                    task = {**task, **task_formulation, "seed": seed, "task_number": i}
                    print("TASK: " + task["mission"])
                    env_string = task["env"]
                    llm_actor = LlmActor(env_string, T, seed=seed)
                    llm_actor.start_render()
                    summary = llm_actor.execute_task(task, step_logging_list=step_logging)
                    llm_actor.stop_render()
                    data.append(summary)
                    pprint.pp(summary)
                except Exception as e:
                    print(f"ERROR {e} for: {task}")
                    raise e

    # save data as df
    df = pd.DataFrame(data)
    df_steps = pd.DataFrame(step_logging)
    current_file_path = os.path.abspath(__file__)
    directory = os.path.join(os.path.dirname(current_file_path), 'analysis', 'data')
    file_path_main = os.path.join(directory, f'{LLM_CONFIG["model"]}_data_main.csv')
    file_path_steps = os.path.join(directory, f'{LLM_CONFIG["model"]}_data_main_steps.csv')
    df.to_csv(file_path_main, index=False)
    df_steps.to_csv(file_path_steps, index=False)
    
