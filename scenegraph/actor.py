import torch
from mini_behavior.minibehavior import MiniBehaviorEnv
from mini_behavior.envs import CleaningUpTheKitchenOnlyEnv

class Actor:
    def __init__(self, env):
        self.env = env
        self.unary_actions = {"pick": self._pick}
        self.binary_actions = {}

        self.reward = None
        self.task_state = None
        
        self.actions = MiniBehaviorEnv.Actions

        self.status_list = ["Success", "Failure", "Type of action cannot be performed", "Couldn't be reached"]
    

    def get_actions(self):
        return [key for key in {**self.unary_actions, **self.binary_actions}.keys()]

    def get_status_list(self):
        return self.status_list

    
    def act(self, action, object_1, object_2=None):
        if action in self.unary_actions:
            status = self.unary_actions[action](object_1)
            return self.vectorize_output(status)
        elif action in self.binary_actions:
            assert object_2 is not None
            status = self.binary_actions[action](object_1, object_2)
            return self.vectorize_output(status)
        else:
            raise NotImplementedError(f"Action: {action} is not implemented. Please choose one of {self.get_actions()}")

    def _pick(self, object_1):
        # assert that object can be picked up
        if "pickup" not in object_1.actions:
            return 2
        # locate object in env
        target_position = object_1.cur_pos

        # move there
        offsets = [
            ((1,0),2), # standing to right looking left
            ((-1,0),0),
            ((0,1),1),
            ((0,-1),3),
        ]
        reached_target = False
        for (offset, orientation) in offsets:
            attempted_position = (target_position[0] + offset[0], target_position[1] + offset[1])
            reached_target = self.navigate_to(attempted_position, orientation)
            if reached_target:
                break
        if not reached_target:
            return 3 # couldn't be reached

        # pick object
        _ = self.env.gen_obs()
        if object_1.inside_of is not None:
            self.env.step(self.actions.open)
            _ = self.env.gen_obs()

        assert self.env.mode == "primitive"
        self.env.step(self.actions.pickup_0)

        return 0

    def vectorize_output(self, status):
        len_status = len(self.status_list)
        output = torch.zeros(len_status)
        output[status] = 1
        return output
    
    def navigate_to(self, target_position, target_orientation):
        try:
            self.env.env.agent_pos = target_position
        except IndexError as e:
            print(e)
            return False
        if not self.env.agent_pos == target_position:
            return False
        while self.env.agent_dir != target_orientation:
            self.env.step(self.actions.right) # turn right
        return True

