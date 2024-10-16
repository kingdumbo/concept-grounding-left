from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import transformers
import torch

from pathlib import Path
import json
import os
import sys

BASEPATH = Path(__file__).resolve().parent
PROMPTS_DIRECTORY = BASEPATH / "prompts"
RESULTS_DIRECTORY = BASEPATH / "results" / "raw"

sys.path.append(BASEPATH)
import utils
from scenegraph import Scenegraph
from utils import correct_parsing, extract_functions_from_domain
from mini_behavior.envs.cleaning_up_the_kitchen_only import CleaningUpTheKitchenOnlyEnv


CONFIG = {
    "model":"gpt-4o-mini",
    "max_tokens": 256,
    "temperature": 0.2,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "response_format": {"type": "text"}
}
class PromptingOpenAI:
    def __init__(self, high_level_prompt_filename, initial_prompt_filename, second_prompt_template_filename, scenegraph, config=CONFIG,):
        self.sg = scenegraph
        self.domain = self.sg.get_domain()
        load_dotenv(find_dotenv())
        self.client = OpenAI()
        self.config = config

        # load prompts
        self.high_level_prompt = utils.load_system_prompt(high_level_prompt_filename)
        self.initial_prompt = utils.load_system_prompt(initial_prompt_filename)
        self.second_prompt_template = utils.load_system_prompt(second_prompt_template_filename)

        # high level prompt needs to maintain history:
        self.history = None

        # for correcting parsings
        self.domain_funcs = extract_functions_from_domain(self.domain)

    def _execute(self, messages, config=None, big_guns=False, seed=None):
        if config is None:
            config = self.config

        if seed is not None:
            config["seed"] = seed

        model = config["model"]
        if big_guns:
            model = "gpt-4o"

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            frequency_penalty=config["frequency_penalty"],
            presence_penalty=config["presence_penalty"],
            response_format=config["response_format"]
        )
        return response

    def to_raw_parsing(self, message, seed=None, apply_correction=False):
        # transform every question in system prompt - user prompt pair
        messages = [
                    {"role": "system", "content": self.initial_prompt},
                    {"role": "user", "content": f"<text>{message}</text>"}
        ]

        # first query produces simplified output
        response = self._execute(messages, seed=seed)

        # then get the programm
        simplified = utils.extract_tag_inner(response.choices[0].message.content, "simplified")[0]
        simplified = f"<simplified>{simplified}</simplified>"
        top_funcs = utils.get_n_likeliest_funcs(simplified, self.domain, n=10)
        modified_system_prompt = self.second_prompt_template.format(top_funcs)
        follow_up_messages = [
                {"role": "system", "content": modified_system_prompt},
                {"role": "user", "content": f"<simplified>{simplified}</simplified>"}
        ]
        
        # append new messages to history
        messages.extend(follow_up_messages)

        # get code
        response = self._execute(messages, seed=seed)

        # extract and return
        raw_parsing = utils.extract_raw_parsing(response.choices[0].message.content)
        
        # apply fixing to raw_parsing
        if apply_correction:
            raw_parsing = correct_parsing(raw_parsing, self.domain_funcs)

        return raw_parsing

    def generate_plan(self, high_level_task, seed=None):
        # generate a scene summary to include with the prompt
        scene_summary = self.sg.get_summary()
        high_level_prompt_modified = self.high_level_prompt.format(scene_summary)
        
        # include system prompt with high level prompt
        messages = [
                    {"role": "system", "content": high_level_prompt_modified},
                    {"role": "user", "content": f"<task>{high_level_task}</task>"}
        ]

        # get initial plan
        response = self._execute(messages, big_guns=True, seed=seed)
        self.history = response

        # extract and return plan as list
        plan = self._extract_plan(response)

        return plan

    def update_plan(self, error_step, output):
        # build update prompt
        update_prompt = f'''An error was encountered by the robot executing the plan!
        Step: {error_step}
        Output: {output}'''
        update_message = [{"role": "user", "content": update_prompt}]
        self.history.extend(update_message)

        # extract updated plan
        response = self._execute(self.history)
        self.history = response

        plan = self._extract_plan(response)

        return plan
        
    def _extract_plan(self, response):
        plans = utils.extract_tag_inner(response.choices[0].message.content, "step")
        return plans


def main():
    # preprocesses
    utils.preprocess()

    # load questionbanks
    questionbank_raw_name = "questionbank_raw.json"
    questionbank_processed_name = "questionbank_processed.json"
    questionbank = utils.load_questionbank(questionbank_raw_name)
    
    # get domain for grounding the prompt
    env = CleaningUpTheKitchenOnlyEnv()
    sg = Scenegraph(env)
    domain = sg.get_domain()
    local_testing = False

    # local_testing = True
    if local_testing:
        questionbank = [questionbank[0]]

    # init client
    config = {
        "model":"gpt-4o-mini",
        "max_tokens": 256,
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": {"type": "text"}
    }

    llm_client = PromptingOpenAI(
        initial_prompt_filename="system_prompt_2_1.txt",
        second_prompt_template_filename="system_prompt_2_2_template.txt",
        config=config,
        domain=domain
    )

    # update question bank with raw parsing per question
    updated_questionbank = [{**qb, "raw_parsing": llm_client.to_raw_parsing(qb["question"])} for qb in questionbank]
    
    # save result
    utils.save_questionbank(updated_questionbank, questionbank_processed_name)

    # and print the result
    print("-------------------\nQUESTIONBANK:\n-------------------")
    print(utils.load_questionbank("questionbank_processed.json"))

if __name__ == "__main__":
    main()
