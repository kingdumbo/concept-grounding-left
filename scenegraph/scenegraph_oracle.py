import sys
from types import NotImplementedType
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the necessary directories to sys.path
sys.path.append(os.path.join(current_dir, '..', 'Jacinle'))
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(current_dir)

from scenegraph import Scenegraph
from utils import load_questionbank
from left.models.model import LeftModel
from typing import Optional, Union, List, Dict
from left.models.reasoning.reasoning import NCOneTimeComputingGrounding
from left.generalized_fol_executor import NCGeneralizedFOLExecutor
from left.generalized_fol_parser import NCGeneralizedFOLPythonParser
import pandas as pd

from mini_behavior.envs import CleaningUpTheKitchenOnlyEnv
class ScenegraphOracle(LeftModel): 
    def __init__(self, domain, scenegraph):

        self.training = False
        self.grounding_cls = scenegraph
        self.domain = domain

        self.parser = NCGeneralizedFOLPythonParser(self.domain, inplace_definition=False, inplace_polymorphic_function=True, inplace_definition_type=False)
        self.executor = NCGeneralizedFOLExecutor(self.domain, self.parser)

    def tell(self, qa_list):
        outputs = {}
        for qa in qa_list:
            self.execute_program_from_parsing_string(
                    question=qa["question"],
                    raw_parsing=qa["raw_parsing"],
                    outputs=outputs,
                    grounding=self.grounding_cls)
        results = [r[2] for r in outputs["results"]]
        answers = [q["answer"] for q in qa_list]
        
        questions = [q["question"] for q in qa_list]
        self._get_pred_answers(outputs, results)
        outputs["answer"] = answers
        outputs["question"] = questions
        return outputs

    def _get_pred_answers(self, outputs_dict, results):
        pred_answers = []
        for result in results:
            answer = self._get_answer(result)
            pred_answers.append(answer)
        outputs_dict["pred_answer"] = pred_answers


    def _get_answer(self, result):
        try:
            result_typename = result.dtype.typename
        except:
            if result is None:
                return ""
            result_typename = result.dtype
        if result_typename == "bool":
            return "yes" if result.tensor.item() > 0 else "no"
        elif result_typename == "int64":
            return str(int(result.tensor.round().item()))
        elif result_typename == "Action":
            idx = result.tensor.argmax()
            return self.grounding_cls.lookup_action_status(idx)
        elif self.grounding_cls.is_descriptor(result_typename):
            breakpoint()
            idx = result.tensor.argmax()
            return self.grounding_cls.lookup_descriptor(result_typename, idx)
        else:
            raise NotImplementedType(f"Unknown question type: {result_typename}")


    def get_accuracy(self, output_dict):
        assert len(output_dict["answer"]) == len(output_dict["pred_answer"])
        equal_entries = sum(1 for a,b in zip(output_dict["answer"], output_dict["pred_answer"]) if a ==b)
        total_entries = len(output_dict["answer"])
        return equal_entries / total_entries

    def summarize_output(self, output_dict, save_to_csv, only_false_answers=True):
        # Unpacking the lists from the dictionary
        questions = output_dict['question']
        pred_answers = output_dict['pred_answer']
        answers = output_dict['answer']
        results = output_dict['results']

        # Preparing lists to store the filtered data
        filtered_questions = []
        filtered_raw_parsing = []
        filtered_answers = []
        filtered_pred_answers = []

        # Iterating through the data
        for question, pred_answer, answer, result in zip(questions, pred_answers, answers, results):
            if not only_false_answers or pred_answer != answer:
                raw_parsing = result[0]  # Extracting raw_parsing from the tuple
                filtered_questions.append(question)
                filtered_raw_parsing.append(raw_parsing)
                filtered_answers.append(answer)
                filtered_pred_answers.append(pred_answer)

        # Creating the DataFrame
        data = {
            "question": filtered_questions,
            "raw_parsing": filtered_raw_parsing,
            "answer": filtered_answers,
            "pred_answer": filtered_pred_answers
        }

        df = pd.DataFrame(data)
        
        # save if necessary
        if save_to_csv:
            df.to_csv("results.csv", index=True)

        return df
        

if __name__ == "__main__":
    env = CleaningUpTheKitchenOnlyEnv()
    sg = Scenegraph(env)
    sg.update()
    #sg.render()
    domain = sg.get_domain()
    domain.print_summary()

    # load processed question bank
    from pathlib import Path
    qb_filepath = Path(__file__).resolve().parent / "questionbank" / "questionbank_debug_fol.json"
    questionbank = load_questionbank(qb_filepath)

    oracle = ScenegraphOracle(domain, sg)
    output = oracle.tell(questionbank)
    acc = oracle.get_accuracy(output)
    print(acc)
    print(oracle.summarize_output(output, save_to_csv=True, only_false_answers=False).head())

    while True:
        raw_parsing=input()
        qa = [{"raw_parsing": raw_parsing, "question":"", "answer": ""}]
        output = oracle.tell(qa)
        print(oracle.summarize_output(output, save_to_csv=False, only_false_answers=False).head())
