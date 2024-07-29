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

def batch_parse_n_save(prompts, filenames, max_gen_len, temperature, top_p):
    results = self.batch_parse(prompts, max_gen_len, temperature, top_p)
    results = [json.dumps(result) for result in results]
    utils.write_text_files(results, filenames, RESULTS_DIRECTORY)
    


def batch_parse(self, prompts, max_gen_len=64, temperature=0.6,top_p=0.9):
    results = self.generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    return results


def main():
    if os.getenv("HF_HOME"):
        model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
        local_testing=False

    else:
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        local_testing=True

    print(f"USING: {model_id}")
    print(f"HF_HOME: {os.getenv('HF_HOME')}")
    print(f"LOCAL TESTING: {local_testing}")

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    utils.preprocess()

    # load prompt files
    initial_prompt = utils.load_system_prompt("system_prompt_2_1.txt")
    second_prompt = utils.load_system_prompt("system_prompt_2_2.txt")
    questionbank = utils.load_questionbank("questionbank_raw.json")
    
    if local_testing:
        questionbank = [questionbank[0]]

    # transform every question in system prompt - user prompt pair
    message_sets = [
        [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"<text>{qb['question']}</text>"}
        ] for qb in questionbank]

    raw_outputs = []
    for i, messages in enumerate(message_sets):
        # first query produces simplified output
        response = pipeline(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        # then get the programm
        modified_system_prompt_2 = second_prompt
        simplified = utils.extract_tag_inner(response[0]["generated_text"][-1]["content"], "simplified")
        simplified = f"<simplified>{simplified}</simplified>"

        
        # append new messages to history
        messages = response[0]["generated_text"]
        messages.extend((modified_system_prompt_2, simplified))

        results = pipeline(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
        raw_outputs.append(results[0]["generated_text"][-1]["content"])
        print(f"TASK: {messages[1]['content']}\nSOLUTION: {results[0]['generated_text'][-1]['content']}")
        print(results)

    # extract answers and write to questionbank
    raw_parsings = utils.extract_raw_parsings(raw_outputs)
    utils.raw_parsings_to_questionbank(questionbank, raw_parsings) 
    
    # and print the result
    print("-------------------\nQUESTIONBANK:\n-------------------")
    print(utils.load_questionbank("questionbank_processed.json"))

if __name__ == "__main__":
    main()
