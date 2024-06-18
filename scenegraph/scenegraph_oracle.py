import sys
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

    def tell(self, raw_parsing:str):
        outputs = {}
        self.execute_program_from_parsing_string(question="",raw_parsing=raw_parsing,outputs=outputs, grounding=self.grounding_cls)
        return outputs


if __name__ == "__main__":
    env = CleaningUpTheKitchenOnlyEnv()
    sg = Scenegraph(env)
    sg.update()
    sg.render()
    domain = sg.get_domain()
    domain.print_summary()
    #raw_parsing = "exists(Object, lambda x: infovofrobot(x))"
    #raw_parsing = "count(Object, lambda x: infovofrobot(x))"
    raw_parsing = "iota(Object, lambda x: infovofrobot(x))"
    oracle = ScenegraphOracle(domain, sg)
    result = oracle.tell(raw_parsing)
    print(result)
