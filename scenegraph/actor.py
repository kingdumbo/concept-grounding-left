import torch
from mini_behavior.minibehavior import MiniBehaviorEnv
from mini_behavior.envs import CleaningUpTheKitchenOnlyEnv

class Actor:
    def __init__(self, env):
        self.env = env
        self.unary_actions = {"pick": self._pick, "place": self._place}
        self.binary_actions = {}

        self.reward = None
        self.task_state = None
        
        self.actions = MiniBehaviorEnv.Actions

        self.status_list = ["Success", "Failure", "Type of action cannot be performed", "Couldn't be reached", "Nothing to place"]

    def get_actions(self):
        return [key for key in {**self.unary_actions, **self.binary_actions}.keys()]

    def get_status_list(self):
        return self.status_list

    
    def act(self, action, object_1, object_2=None):
        if action in self.unary_actions:
            status, reward, done = self.unary_actions[action](object_1)
            return self.vectorize_output(status), reward, done
        elif action in self.binary_actions:
            assert object_2 is not None
            status, reward, done = self.binary_actions[action](object_1, object_2)
            return self.vectorize_output(status), reward, done
        else:
            raise NotImplementedError(f"Action: {action} is not implemented. Please choose one of {self.get_actions()}")

    def _pick(self, object_1):
        # assert that object can be picked up
        if "pickup" not in object_1.actions:
            return 2
        # locate object in env
        target_position = object_1.cur_pos

        # move there
        reached_target = self.navigate_to(target_position, look_at=True)
        if not reached_target:
            return 3 # not reached

        # pick object
        _ = self.env.gen_obs()
        if object_1.inside_of is not None:
            self.env.step(self.actions.open)

        assert self.env.mode == "primitive"

        # try different pick actions until object is carried
        pick_actions = [self.actions.pickup_0, self.actions.pickup_1, self.actions.pickup_2]
        for action in pick_actions:
            _, reward, done, _ = self.env.step(action)
            if object_1 in self.env.carrying:
                return 0, reward, done

        return 1, reward, done

    def _place(self, object_1):
        # assert that there is something to place
        something_to_place = False
        for obj in self.env.carrying:
            if "drop" in obj.actions or "dropin" in obj.actions:
                something_to_place = True
                break
        if not something_to_place:
            return 4 # nothing to place

        # locate targe-object in env
        target_position = object_1.cur_pos

        # move there
        reached_target = self.navigate_to(target_position, look_at=True)
        if not reached_target:
            return 3 # not reached

        # place object
        _ = self.env.gen_obs()
        # target may havbe to be opened first
        if "open" in object_1.actions:
            self.env.step(self.actions.open)

        assert self.env.mode == "primitive"
        carrying_before = [obj for obj in self.env.carrying]
        place_actions = [self.actions.drop_0, self.actions.drop_1, self.actions.drop_2, self.actions.drop_in]
        for action in place_actions:
            _, reward, done, _ = self.env.step(action)
            for obj in carrying_before:
                if obj.inside_of:
                    return 0, reward, done

        return 1, reward, done


    def vectorize_output(self, status):
        len_status = len(self.status_list)
        output = torch.zeros(len_status)
        output[status] = 1
        return output
    
    def navigate_to(self, target_position, look_at=True):
        offsets = [
            ((1,0),2), # standing to right looking left
            ((-1,0),0),
            ((0,1),1),
            ((0,-1),3),
        ]
        for (offset, orientation) in offsets:
            if look_at:
                attempted_position = (target_position[0] + offset[0], target_position[1] + offset[1])
            else:
                attempted_position = target_position
            try:
                self.env.env.agent_pos = attempted_position
            except IndexError as e:
                print(e)
                return False
            if self.env.agent_pos == attempted_position:
                while self.env.agent_dir != orientation:
                    self.env.step(self.actions.right) # turn righattempted_position
                return True
        return False

