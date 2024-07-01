import fire
from llama import Llama

from pathlib import Path
import json
import os
import sys

BASEPATH = Path(__file__).resolve().parent
PROMPTS_DIRECTORY = BASEPATH / "prompts"
RESULTS_DIRECTORY = BASEPATH / "results" / "raw"

sys.path.append(BASEPATH)
import utils


class Llama3Parser:
    def __init__(self, ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
        
        # instantiate llama
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )


    def batch_parse_n_save(self, prompts, filenames, max_gen_len, temperature, top_p):
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


def main(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size, max_gen_len, temperature, top_p):
    # load questions and save as prompt-files (txt)
    utils.preprocess()

    # load prompt files
    prompts, filenames = utils.load_text_files(PROMPTS_DIRECTORY)

    # parse them with the llm and save
    llm = Llama3Parser(
        ckpt_dir=ckpt_dir, 
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    llm.batch_parse_n_save(prompts, filenames, max_gen_len=max_gen_len, temperature=temperature,top_p=top_p)

    # update the questionbank with the raw-parsing translation of the questions
    utils.process_llm_output()

    # and print the result
    print(utils.load_questionbank(BASEPATH / "questionbank" / "questionbank_processed.json"))

if __name__ == "__main__":
    fire.Fire(main)

