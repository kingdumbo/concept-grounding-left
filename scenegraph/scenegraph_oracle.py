import sys
from types import NotImplementedType
sys.path.append("/home/max/uni/LEFT/Jacinle")
sys.path.append("/home/max/uni/LEFT/")
sys.path.append("home/max/uni/LEFT/scenegraph")

from scenegraph import Scenegraph
from left.models.model import LeftModel
from typing import Optional, Union, List, Dict
from left.models.reasoning.reasoning import NCOneTimeComputingGrounding
from left.generalized_fol_executor import NCGeneralizedFOLExecutor
from left.generalized_fol_parser import NCGeneralizedFOLPythonParser

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
        q_types = [q["type"] for q in qa_list]
        results = [r[2] for r in outputs["results"]]
        answers = [q["answer"] for q in qa_list]
        self._get_pred_answers(outputs, results, q_types)
        outputs["answer"] = answers
        return outputs

    def _get_pred_answers(self, outputs_dict, results, q_types):
        pred_answers = []
        for result, q_type in zip(results, q_types):
            answer = self._get_answer(result, q_type)
            pred_answers.append(answer)
        outputs_dict["pred_answer"] = pred_answers


    def _get_answer(self, result, q_type):
        if q_type == "bool":
            return "yes" if result.tensor.item() > 0 else "no"
        elif q_type == "int64":
            return str(int(result.tensor.round().item()))
        else:
            raise NotImplementedType(f"Unknown question type: {q_type}")


    def get_accuracy(self, output_dict):
        assert len(output_dict["answer"]) == len(output_dict["pred_answer"])
        equal_entries = sum(1 for a,b in zip(output_dict["answer"], output_dict["pred_answer"]) if a ==b)
        total_entries = len(output_dict["answer"])
        return equal_entries / total_entries


if __name__ == "__main__":
    env = CleaningUpTheKitchenOnlyEnv()
    sg = Scenegraph(env)
    sg.update()
    #sg.render()
    domain = sg.get_domain()
    #domain.print_summary()

    ground_truth = [
        {
            "question": "How many objects are in front of the robot?",
            "raw_parsing":  "count(Object, lambda x: infovofrobot(x))",
            "answer": "4",
            "type": "int64"
        },
        {
            "question": "Is there an object in front of the robot, which is also next to another object?",
            "raw_parsing": "exists(Object, lambda x: infovofrobot(x) and exists(Object, lambda y: nextto(x, y)))",
            "answer": "yes",
            "type": "bool"
        }
    ]

    oracle = ScenegraphOracle(domain, sg)
    # Interactive loop to accept arguments and call get_attr_for_id
    #while True:
    #    user_input = input("Enter raw parsing or 'q' to quit: ")
    #    if user_input.strip().lower() == 'q':
    #        break
    #    raw_parsing = user_input
    #    result = oracle.tell(raw_parsing)
    #    output = result["results"][0][2]
    #    print(output)
    output = oracle.tell(ground_truth)
    acc = oracle.get_accuracy(output)
    print(acc)
