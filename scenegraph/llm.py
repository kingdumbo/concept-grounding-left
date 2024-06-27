#from llama import Llama

from pathlib import Path
import os

BASEBATH = Path(__file__).resolve().parent
PROMPTS_DIRECTORY = BASEBATH / "prompts"
RESULTS_DIRECTORY = BASEBATH / "results" / "raw"

def load_text_files(directory):
    text_list = []
    filenames = []
    for filepath in directory.glob("*.txt"):
        with filepath.open('r') as file:
            text_list.append(file.read())
            filenames.append(filepath.name)
    return text_list, filenames

def write_text_files(text_list, filenames, directory):
    for text, filename in zip(text_list, filenames):
        output_path = directory / filename
        with output_path.open('w') as file:
            file.write(text)

class Llama3Parser:
    def __init__(self, ckpt_dir, tokenizer_path):
        
        # instantiate llama
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=8192,
            max_batch_size=100,
        )


    def batch_parse_n_save(self, prompts, filenames):
        results = self.batch_parse(prompts, filenames)
        write_text_files(results, filenames, RESULTS_DIRECTORY)
        


    def batch_parse(self, prompts, max_gen_len=1000, temperature=0.6,top_p=0.9):
        results = self.generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        return results


if __name__ == "__main__":

    # get paths from environment variables
    ckpt_dir = os.getenv("CKPT_DIR", "/work/projects/project02201/Sachin_ws/llama3/Meta-Llama-3-70B-Instruct")
    tokenizer_path = os.getenv("TOKENIZER_PATH", "/work/projects/project02201/Sachin_ws/llama3/Meta-Llama-3-70B-Instruct/tokenizer.model")

    # load prompts
    prompts, filenames = load_text_files(PROMPTS_DIRECTORY)
    llm = Llama3Parser(
        ckpt_dir=ckpt_dir, 
        tokenizer_path=tokenizer_path
    )
    llm.batch_parse_n_save(prompts, filenames)
