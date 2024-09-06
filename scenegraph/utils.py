from pathlib import Path
import re
import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import socket
import torch

BASEBATH = Path(__file__).resolve().parent
PROMPTS_DIRECTORY = BASEBATH / "prompts"
RESULTS_DIRECTORY = BASEBATH / "results" / "raw"
QUESTIONBANK_DIRECTORY = BASEBATH / "questionbank"

def load_text_files(directory):
    text_list = []
    filenames = []
    textfiles_sorted = sorted(directory.glob("*.txt"))
    for filepath in textfiles_sorted:
        with filepath.open('r') as file:
            text_list.append(file.read())
            filenames.append(filepath.name)
    return text_list, filenames

def write_text_files(text_list, filenames, directory):
    for text, filename in zip(text_list, filenames):
        output_path = directory / filename
        with output_path.open('w') as file:
            file.write(text)


def extract_tag_inner(text, tag):
    """
    Extract all strings between the specified tags in the input text.

    Args:
    - input_text (str): The text to search within.
    - tag (str): The tag name to search for.

    Returns:
    - List[str]: A list of strings found between the specified tags.
    """
    # Create a regular expression pattern for the specified tag
    pattern = fr'<{tag}>(.*?)</{tag}>'
    
    # Find all strings between the specified tags
    extracted_strings = re.findall(pattern, text, re.DOTALL)
    
    return extracted_strings

def build_prompt_files(questionbank, template_file_directory, template_file_name, prompt_file_directory):
    # for every prompt, create txt file in prompts directory
    template_file_path = template_file_directory / template_file_name
    with open(template_file_path, "r") as f:
        prompt_template = f.read()

    # iterate over questions and prompt to file
    for i, qa in enumerate(questionbank):
        question = qa["question"]
        prompt = prompt_template.format(question=question)
        filename = f"{i}.txt"
        full_path = prompt_file_directory / filename
        with open(full_path, "w") as f:
            f.write(prompt)

def extract_raw_parsings(raw_llm_output_strings):
    raw_parsings = []
    for i, output in enumerate(raw_llm_output_strings):
        raw_parsings.append(extract_raw_parsing(output))
    return raw_parsings

def extract_raw_parsing(output):
    raw_parsing = ""
    try:
        raw_parsing = extract_tag_inner(output, "code")[0]
    except Exception as e:
        print(f"No code found at id {i} for raw output: {output}")
    return raw_parsing

def load_questionbank(filename):
    # return questionbank as object ready for scenegraph oracle
    path = QUESTIONBANK_DIRECTORY / filename
    d = []
    with open(path, "r") as f:
        print(path)
        d = json.load(f)
    return d

def save_questionbank(questionbank, filename="questionbank_processed.json"):
    # write
    target_path = QUESTIONBANK_DIRECTORY / filename
    with open (target_path, "w") as f:
        json.dump(questionbank, f, indent=4)


def raw_parsings_to_questionbank(questionbank, raw_parsings, filename="questionbank_processed.json"):
    # write the raw-parsings to the updated question bank
    for i in range(len(questionbank)):
        raw_parsing = raw_parsings[i]
        questionbank[i]["raw_parsing"] = raw_parsing
    save_questionbank(questionbank, filename=filename)


def preprocess(raw_qb_filename = "questionbank_raw.json"):
    # load questionbank questions and save individually as textfiles with prompt
    qb_filepath = QUESTIONBANK_DIRECTORY / raw_qb_filename
    questionbank = load_questionbank(qb_filepath)

    prompt_template_dir = PROMPTS_DIRECTORY / "templates" 
    prompt_template = "base_with_modifiedexamples.txt"
    build_prompt_files(questionbank, prompt_template_dir, prompt_template, PROMPTS_DIRECTORY) 


def process_llm_output(raw_qb_filename = "questionbank_raw.json"):
    # load all response files and use them to update the questionbank
    qb_filepath = QUESTIONBANK_DIRECTORY / raw_qb_filename
    questionbank = load_questionbank(qb_filepath)
    raw_outputs, _ = load_text_files(RESULTS_DIRECTORY)
    raw_parsings = extract_raw_parsings(raw_outputs)
    raw_parsings_to_questionbank(questionbank, raw_parsings, QUESTIONBANK_DIRECTORY)


def load_system_prompt(filename):
    # return system prompt template as string
    path = PROMPTS_DIRECTORY / "templates" / filename
    with open(path, "r") as f:
        return f.read()
    raise Exception()

def extract_functions_from_string(raw, keep_first=False):
    # Regular expression to find all sequences of letters longer than 1
    pattern = r'\b[a-zA-Z]{2,}\b'
    words = re.findall(pattern, raw) 
    
    # remove first word (describe_count, ... = built-int) and lambda 
    blacklist = ["lambda", "iota", "and", "or", "not", "exists"]
    idx = 1
    if keep_first:
        idx = 0
    functions = [word for word in words[idx:] if word not in blacklist]

    return functions

def extract_functions_from_domain(domain):
    funcs = []
    for type in domain.types.values():
        funcs.append(str(type))
    for const in domain.constants.values():
        funcs.append(str(const))
    for function in domain.functions.values():
        funcs.append(function.name)
    # remove duplicates
    funcs = list(set(funcs))
    return funcs

def correct_parsing(raw_parsing, domain_funcs, verbose=True):
    parsing_funcs = extract_functions_from_string(raw_parsing)
    unreplaceable = []
    corrected_raw_parsing = raw_parsing
    for func in parsing_funcs:
        # if already correct, do nothing
        if func in domain_funcs:
            continue
        # else find closest replacement and replace
        best_match = process.extractOne(func, domain_funcs, scorer=fuzz.token_sort_ratio)
        if not best_match:
            unreplaceable.append(func)
        else:
            # replace in string
            best_match = best_match[0]
            pattern = fr"\b{func}\b"
            corrected_raw_parsing = re.sub(pattern, best_match, corrected_raw_parsing)
    if verbose and len(unreplaceable) > 0:
        print(f"NO CORRECTION for {unreplaceable} IN: {raw_parsing}")
    return corrected_raw_parsing


def correct_parsings(raw_parsings, domain, verbose=True):
    # extract all from domain
    domain_funcs = extract_functions_from_domain(domain)
    
    corrected_parsings = [correct_parsing(raw, domain_funcs, verbose=verbose) for raw in raw_parsings]

    return corrected_parsings

def get_n_likeliest_funcs(simplified, domain, n):
    # get func and signature
    all_funcs = {}
    for function in domain.functions.values():
        name = function.name
        nr_arguments = function.nr_arguments
        all_funcs[name] = nr_arguments

    # Store scores
    scores = []
    
    for word in simplified.split():
        for func_name in all_funcs.keys():
            score = fuzz.ratio(word, func_name)
            scores.append((func_name, score))
    
    # Sort by score in descending order
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Remove duplicates, keeping the one with the highest score
    seen = set()
    unique_scores = []
    for func_name, score in scores:
        if func_name not in seen:
            seen.add(func_name)
            unique_scores.append((func_name, score))
    
    # Get function signatures
    return_str = ""
    for func, _ in unique_scores[:n]:
        signature = "x" if all_funcs[func] == 1 else "x,y"
        func_str = f"{func}({signature})\n"
        return_str += func_str

    return(return_str)


def check_port(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)  # Set a timeout for the connection attempt
    try:
        sock.connect((host, port))
        sock.close()
        return True  # Port is in use
    except (socket.timeout, ConnectionRefusedError):
        return False  # Port is free or connection refused

def check_min_max_diff(tensor_val, th):
    min_val = torch.min(tensor_val.tensor)
    max_val = torch.max(tensor_val.tensor)
    difference = max_val - min_val
    return bool(difference > th)


if __name__ == "__main__":
    # imports
    import sys
    sys.path.append("/home/max/uni/LEFT/scenegraph/")
    from scenegraph import Scenegraph
    from mini_behavior.envs.cleaning_up_the_kitchen_only import CleaningUpTheKitchenOnlyEnv

    # get domain
    env = CleaningUpTheKitchenOnlyEnv()
    sg = Scenegraph(env)
    domain = sg.get_domain()

    print(load_system_prompt("system_prompt_2_1.txt"))
    template_prompt = load_system_prompt("system_prompt_2_2_template.txt")
    simplified = "exists rag on top of countertop"
    top_funcs = print_n_likeliest_funcs(simplified, domain, n=10)
    modified_system_prompt = template_prompt.format(top_funcs)
    print(modified_system_prompt)

    ## get questionbank
    #qb = load_questionbank("questionbank_processed.json")
    #raw_parsings = [q["raw_parsing"] for q in qb]

    ## get domain funcs
    #domain_funcs = extract_functions_from_domain(domain)

    ## correct and update questionbank
    #for q in qb:
    #    q["raw_parsing"] = correct_parsing(q["raw_parsing"], domain_funcs)

    ## update
    #save_questionbank(qb, "questionbank_corrected.json")



