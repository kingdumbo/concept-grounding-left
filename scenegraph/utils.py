from pathlib import Path
import re
import json

BASEBATH = Path(__file__).resolve().parent
PROMPTS_DIRECTORY = BASEBATH / "prompts"
RESULTS_DIRECTORY = BASEBATH / "results" / "raw"
QUESTIONBANK_DIRECTORY = BASEBATH / "questionbank"

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
    extracted_strings = re.findall(pattern, input_text, re.DOTALL)
    
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
        raw_parsing = ""
        try:
            raw_parsing = extrag_tag_inner(output, "code")[0]
        except Exception as e:
            print(f"No code found at id {i} for raw output: {output}")
        raw_parsings.append(raw_parsing)

    return raw_parsings


def load_questionbank(path):
    # return questionbank as object ready for scenegraph oracle
    d = []
    with open(path, "r") as f:
        print(path)
        d = json.load(f)
    return d


def raw_parsings_to_questionbank(questionbank, raw_parsings, target_directory):
    # write the raw-parsings to the updated question bank
    for i in range(len(questionbank)):
        raw_parsing = raw_parsings[i]
        questionbank[i]["raw_parsing"] = raw_parsing

    # write
    target_path = target_directory / "questionbank_processed.json"
    with open (target_path, "w") as f:
        json.dump(questionbank, f, indent=4)

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

if __name__ == "__main__":
    preprocess()
    #
    # HERE WOULD HAPPEN THE LLM MAGIC
    #
    process_llm_output()

    # for visualization, load and print
    print(load_questionbank(QUESTIONBANK_DIRECTORY / "questionbank_processed.json"))

