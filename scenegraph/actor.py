import torch


class Actor:
    def __init__(self, env):
        self.env = env
        self.unary_actions = {"pick": self._pick}
        self.binary_actions = {}

        self.reward = None
        self.task_state = None

        self.status_list = ["Success", "Failure"]
    

    def get_actions(self):
        return [key for key in {**self.unary_actions, **self.binary_actions}.keys()]

    def get_status_list(self):
        return self.status_list

    
    def act(self, action, object_1, object_2=None):
        if action in self.unary_actions:
            return self.unary_actions[action](object_1)
        elif action in self.binary_actions:
            assert object_2 is not None
            return self.binary_actions[action](object_1, object_2)
        else:
            raise NotImplementedError(f"Action: {action} is not implemented. Please choose one of {self.get_actions()}")

    def _pick(self, object_1):
        # locate object in env

        # plan to move there

        # pick object

        status = 0
        return status

    def vectorize_output(status):
        len_status = len(self.status_list)
        output = torch.zeros(len_status)
        output[status] = 1
        return output

