import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the necessary directories to sys.path
sys.path.append(os.path.join(current_dir, '..', 'Jacinle'))
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(current_dir)

from constants import AGENT_RELATIVE_STATES, ABSOLUTE_STATES, RELATIONS, OBJ_BOOL_PROPS, OBJ_PROPS
from actor import Actor
from mini_behavior.envs.cleaning_up_the_kitchen_only import CleaningUpTheKitchenOnlyEnv
from concepts.dsl.dsl_functions import Function, FunctionTyping
from concepts.dsl.dsl_types import ObjectType, BOOL, INT64, Variable
from concepts.dsl.function_domain import FunctionDomain
from left.domain import create_bare_domain
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import torch
from livereload import Server

class Scenegraph:
    def __init__(self, env):
        self.rendering = False

        self.env = env

        # set reference to robot
        self.robot_id = "robot"

        # initialize properties of obj e.g. type="blender" to bool blender(x)->bool
        self.descriptor_store = {} # for storing e.g. Color=red for object 5
        self.bool_props = OBJ_BOOL_PROPS
        self.obj_props = OBJ_PROPS
        all_obj_prop_values = self._get_all_obj_prop_vals(env, self.obj_props)

        # set types of states
        self.agent_rel_states = AGENT_RELATIVE_STATES
        self.abs_states = ABSOLUTE_STATES
        self.attributes = {**self.abs_states, **all_obj_prop_values, **self.bool_props, self.robot_id: self.robot_id}
        # TODO include type in attributes (furniture)
        self.relations = {**RELATIONS, **self.agent_rel_states}


        # to simplify lookup for relational attributes
        self.obj_to_node = {}
        self.node_to_obj = {}
        # keep an exact ordering of the ids for lookup, of obj ids as they're added to the graph
        self.obj_ids = []

        # for rendering
        self.sg_color_map = []

        self.reset() # set empty graph

        # for FOL executor
        self.training = False

        # for actions
        self.actor = Actor(self.env)
        self.mission = self.env.gen_full_obs()["mission"]
        self.current_reward = 0
        self.task_done = False

    def update(self):
        state = self.env.get_state()
        self.reset() # always start from empty graph

        # add node for agent
        self.sg.add_node(self.robot_id)
        self.sg_color_map.append("red")
        self.obj_ids.append(self.robot_id)
        self.objs.append(None)

        # create attribute for robot
        self.set_attr_for_id(self.robot_id, 1, self.robot_id)

        # TODO add agent position and orientation
       
        # FIRST:
        # add node for each object together with absolute properties
        # and agent-relative states
        
        # get objects (including furniture and other objects)
        objs = state["objs"]
        objs = self.env.objs
    
        # iterate over all objects
        for label, obj_list in objs.items():
            for idx, obj in enumerate(obj_list):
#
                # label and index jointly unique
                id = f"{label}_{str(idx)}"
    
                # add for lookup
                self.obj_to_node[obj] = id
                self.node_to_obj[id] = obj
                self.obj_ids.append(id)
                self.objs.append(obj)
    
                # extract absolute and agent-relative attributes
                obj_attributes = {}
                obj_agent_relations = {}

                # extract obj properties
                for prop, alias in self.obj_props.items():
                    val = getattr(obj, prop)
                    # add to graph
                    obj_attributes[val] = True

                    # add to value store 
                    self.set_attr_for_id(val, 1, id)

                    # store for description
                    if self.is_descriptor(prop):
                        self.set_descriptor_for_id(prop, val, id)

                # add boolean props
                for prop, alias in self.bool_props.items():
                    if getattr(obj, prop):
                        # add to graph
                        obj_attributes[alias] = True

                        # add to value store 
                        self.set_attr_for_id(alias, 1, id)


                # check if furniture, if yes, connect with agent
                # and color accordingly
                color = "blue"
                if obj.is_furniture():
                    # add edge between furniture and agent
                    #self.sg.add_edge(id, self.robot_id, near=True)
                    color = "green"

                    # add to value store 
                    #self.set_attr_for_id("near", 1, id, self.robot_id, symmetric=True)

                self.sg_color_map.append(color)
    
    
                for attribute, state in obj.states.items():
                    # check if is agent-relative
                    # NOTE: agent relative attributes are treated as essentially absolute attributes of the object
                    if attribute in self.agent_rel_states:
                        val = state.get_value(self.env)
                        if val:
                            # add for graph
                            alias = self.agent_rel_states[attribute]
                            obj_agent_relations[alias] = True
                            
                            # add to store, but as relation
                            self.set_attr_for_id(alias, 1, id, self.robot_id)
    
                    # otherwise check if absolute attribute
                    elif attribute in self.abs_states:
                        val = state.get_value(self.env)
                        if val:
                            # add for graph
                            alias = self.abs_states[attribute]
                            obj_attributes[alias] = True

                            # add to store
                            self.set_attr_for_id(alias, 1, id)
    
                    # for completeness check if relative, but will be implemented later 
                    elif attribute in self.relations:
                        pass
    
                    else:
                        raise NotImplementedError(f"Unknown attribute: {attribute}")
                    
                # add node with node attributes
                self.sg.add_node(id, **obj_attributes)
    
                # add edge to agent if at least one attribute
                if len(obj_agent_relations.keys()) > 0:
                    self.sg.add_edge(id, self.robot_id, **obj_agent_relations)
    
        # SECOND:
        # add relational attributes between objects
        # again iterate over objects
        for obj1 in self.obj_to_node.keys():
            for obj2 in self.obj_to_node.keys():
    
                # don't eval relations of they are the same object
                if obj1 == obj2:
                    continue
    
                # store (directed relations) for graph
                relations_obj1_obj2 = {}
    
                # evaluate all relational attributes 
                for relation, alias in self.relations.items():
                    try:
                        val = obj1.states[relation].get_value(obj2, self.env)
                        if val:
                            # add for graph
                            alias = self.relations[relation]
                            relations_obj1_obj2[alias] = True

                            # add to store
                            obj_id_from = self.obj_to_node[obj1]
                            obj_id_to = self.obj_to_node[obj2]
                            self.set_attr_for_id(alias, 1, obj_id_from, obj_id_to)
                    except:
                        continue
    
                # add edge if at least one relation exists
                if len(relations_obj1_obj2.keys()) > 0:
                    node1 = self.obj_to_node[obj1]
                    node2 = self.obj_to_node[obj2]
                    self.sg.add_edge(node1, node2, **relations_obj1_obj2)


    def reset(self):
        self.sg = nx.MultiDiGraph() # to allow for multiple edges between two nodes

        # initialize embeddings
        num_env_objects = 1 # because of agent
        # iterate over all objects (stored in lists of multiple objects of the same type!)
        for obj in self.env.objs.values():
            num_env_objects += len(obj)
        num_atts= len(self.attributes) + 1 # because of robot/agent
        num_rels = len(self.relations)
        self.attribute_vals = torch.zeros((num_atts, num_env_objects))
        self.relation_vals = torch.zeros((num_rels, num_env_objects, num_env_objects))

        # reset storage of descriptors
        self.descriptor_store = {}
        _ = self._get_all_obj_prop_vals(self.env, self.obj_props)
        for key in self.descriptor_store.keys():
            num_descriptions = len(self.descriptor_store[key]["descriptions"])
            self.descriptor_store[key]["values"] = torch.zeros((num_env_objects, num_descriptions))

        # reset ordering of ids
        self.obj_ids =  []
        self.objs = []

        # reset lookups
        self.obj_to_node = {}
        self.node_to_obj = {}

        # reset action relevant things
        self.mission = self.env.gen_full_obs()["mission"]
        self.current_reward = 0
        self.tas_done = False

    def id_to_idx(self, id):
        if id in self.obj_ids:
            return self.obj_ids.index(id)
        raise ValueError(f"Unkown node-id: {id}")

    def idx_to_obj(self, idx):
        try:
            return self.objs[idx]
        except IndexError as e:
            raise IndexError(f"Idx {idx} is out of bounds for objects stored in self.objs")


    def attr_to_idx(self, category, name):
        # id of attribute to index on vector of all feature vectors
        # get correct list of attributes to index from
        if not category in ["attribute", "relation"]: raise NotImplementedError("Multi-relation not implemented on graph")
        attribute_dict = getattr(self, f"{category}s")

        # get index if the name of the attribute as index
        attr_aliases = list(attribute_dict.values())
        if not name in attr_aliases: raise ValueError(f"{name} not in list of possible {category}s")

        index = attr_aliases.index(name)

        return index

    def descriptor_to_idx(self, descriptor_category, value):
        # gets the index of e.g. color=red
        descriptor_category = descriptor_category.capitalize()
        if descriptor_category in self.descriptor_store:
            descriptions = self.descriptor_store[descriptor_category]["descriptions"]
            if value in descriptions:
                return descriptions.index(value)
            raise ValueError(f"{value} is not a value that descriptor {descriptor_category} can take!")
        else:
            raise KeyError(f"{descriptor_store} is not a known description category!")


    def set_attr_for_id(self, attribute, value, id, id2=None, id3=None, symmetric=False):
        # set attribute value for specific node id and attribute
        # check what category attribute belongs to
        if id2 is None and id3 is None:
            # if only id is present, must be attribute
            category = "attribute"

            # get required indices
            obj_id = self.id_to_idx(id)
            attr_id = self.attr_to_idx(category, attribute)

            # set value
            self.attribute_vals[attr_id, obj_id] = value

        elif id2 is not None and id3 is None:
            # if id and and id2, must be a relation
            category = "relation"

            # get required indices
            obj_id_from = self.id_to_idx(id)
            obj_id_to = self.id_to_idx(id2)
            attr_id = self.attr_to_idx(category, attribute)

            # set value
            # the first obj is the "from" the second obj is the "to"
            #if the relation implies a directionality
            self.relation_vals[
                attr_id,
                obj_id_from,
                obj_id_to] = value

            if symmetric:
                self.relation_vals[
                    attr_id,
                    obj_id_to,
                    obj_id_from] = value

        elif id2 is not None and id3 is not None:
            raise NotImplementedError("Multi-relation not implemented on graph")

        else:
            # if none applies, somethings wrong
            raise LookupError(f"No category for id={id}, id2={id2}, id3={id3}")

    def set_descriptor_for_id(self, descriptor_category, descriptor_value, id):
        # set color=red for an object identified by id
        descriptor_category = descriptor_category.capitalize()
        obj_idx = self.id_to_idx(id)
        if descriptor_category in self.descriptor_store:
            descriptions = self.descriptor_store[descriptor_category]["descriptions"]
            if descriptor_value in descriptions:
                descript_idx = descriptions.index(descriptor_value)
                self.descriptor_store[descriptor_category]["values"][obj_idx, descript_idx] = 1
            else:
                raise ValueError(f"{descriptor_value} is not in description in category {descriptor_category}")
        else:
            raise KeyError(f"{descriptor_category} is not known!")

        
    def get_attr_for_id(self, attribute, id, id2=None, id3=None):
        # set attribute value for specific node id and attribute
        # check what category attribute belongs to
        if id2 is None and id3 is None:
            # if only id is present, must be attribute
            category = "attribute"

            # get required indices
            obj_id = self.id_to_idx(id)
            attr_id = self.attr_to_idx(category, attribute)

            # get value
            return self.attribute_vals[attr_id, obj_id]

        elif id2 is not None and id3 is None:
            # if id and and id2, must be a relation
            category = "relation"

            # get required indices
            obj_id_from = self.id_to_idx(id)
            obj_id_to = self.id_to_idx(id2)
            attr_id = self.attr_to_idx(category, attribute)

            # get value
            # the first obj is the "from" the second obj is the "to"
            #if the relation implies a directionality
            return self.relation_vals[
                attr_id,
                obj_id_from,
                obj_id_to]

        elif id2 is not None and id3 is not None:
            raise NotImplementedError("Multi-relation not implemented on graph")

        else:
            # if none applies, somethings wrong
            raise LookupError(f"No category for id={id}, id2={id2}, id3={id3}")

    def lookup_descriptor(self, descriptor_category, idx):
        # get the e.g. color at index 7
        descriptor_category = descriptor_category.capitalize()
        if descriptor_category in self.descriptor_store:
            try:
                return self.descriptor_store[descriptor_category]["descriptions"][idx]
            except IndexError as e:
                raise IndexError(f"Index: {idx} doesn't refer to a valid description, is out of bounds")
        else:
            raise KeyError(f"{descriptor_categry} not a known description category!")

    def lookup_action_status(self, idx):
        action_statuses = self.actor.get_status_list()
        try:
            return action_statuses[idx]
        except IndexError as e:
            raise IndexError(f"Index: {idx} doesn't refer to a valid action status, is out of bounds")


    def compute_similarity(self, concept_cat, concept):
        attribute_index = self.attr_to_idx(concept_cat, concept)
        if concept_cat == "attribute":
            feature_vector = self.attribute_vals[attribute_index, :]

        elif concept_cat == "relation":
            feature_vector = self.relation_vals[attribute_index,:,:]
        elif concept_cat == "multi_relation":
            raise NotImplementedError("Multi-relation not implemented on graph")
        else:
            raise NotImplementedError(f"Unknown concept_cat: {concept_cat}")

        return feature_vector

    def compute_description(self, concept_cat, attr_to_be_described):
        if attr_to_be_described in self.descriptor_store:
            return self.descriptor_store[attr_to_be_described]["values"]
        raise KeyError(f"{attr_to_be_described} is not an known categor for description!")


    def compute_action(self, object_tensor_1, object_tensor_2, action_name):
        # check if action is valid
        assert action_name in self.actor.get_actions()

        # get actual object instances to which object_1 and object_2 (if present) refer
        idx = object_tensor_1.tensor.argmax()
        object_1 = self.idx_to_obj(idx)
        object_2 = None
        
        if object_tensor_2 is not None:
            idx_2 = object_tensor_2.tensor.argmax()
            object_2 = self.idx_to_obj(idx_2)

        # execute action with actor
        result, reward, done =  self.actor.act(action_name, object_1, object_2)
        self.current_reward = reward
        self.task_done = done
        return result


    def is_descriptor(self, descriptor):
        if descriptor.capitalize() in self.descriptor_store:
            return True
        if descriptor in self.descriptor_store:
            return True
        return False


    def get_descriptions(self):
        return [key for key in self.descriptor_store.keys()]
        

    def render(self, fancy_vis=True, continual_rendering=False):
        # render graph either using matplotlib or with pyvis (fancy)
        if fancy_vis == False:
            pos = nx.spring_layout(self.sg, k=0.1, iterations=20)
            nx.draw(self.sg, pos=pos, node_color=self.sg_color_map, with_labels=True)
            plt.show()
        else:
            net = Network(directed=True)
            # Customize nodes to include arbitrary attributes in the title
            for (node, data), color in zip(self.sg.nodes(data=True), self.sg_color_map):
                title = " | ".join([f"{key}: {value}" for key, value in data.items()])
                net.add_node(node, title=title, color=color, **data)

            # Customize edges to include arbitrary attributes in the title
            for source, target, data in self.sg.edges(data=True):
                title = " | ".join([f"{key}: {value}" for key, value in data.items()])
                net.add_edge(source, target, title=title, **data)
            if not continual_rendering:
                net.show("network.html", notebook=False)
            else:
                self.html_filename = "network.html"
                # Save the network visualization to a fixed HTML file
                path = "/home/max/uni/LEFT/network.html"
                net.write_html(path, open_browser =False)
                
                if not self.rendering:
                    import threading
                    self.rendering=True
                    server = Server()
                    path = path
                    server.watch(path)
                    server_thread = threading.Thread(target=server.serve)
                    server_thread.setDaemon(True)
                    server_thread.start()


    def get_domain(self):
        # constructs domain from graph/evnv
        domain = create_bare_domain()

        OBJECT = ObjectType("Object")
        ACTION = ObjectType("Action")

        # add ations manually
        domain.define_function(Function("pick", FunctionTyping[BOOL](ACTION, OBJECT)))
        domain.define_function(Function("place", FunctionTyping[BOOL](ACTION, OBJECT)))
        
        # create functions for the attributes
        for attr in self.attributes.values():
            #domain.define_function(Function(f"{attr}_Object", FunctionTyping[BOOL](OBJECT)))
            domain.define_function(Function(attr, FunctionTyping[BOOL](OBJECT)))

        # create functions for the relations
        for rel in self.relations.values():
            domain.define_function(Function(rel, FunctionTyping[BOOL](OBJECT,OBJECT)))

        # Create functions and types for the descriptions
        for descriptor in self.descriptor_store.keys():
            lower_case = descriptor.lower()
            TYPE = ObjectType(descriptor)
            domain.define_type(TYPE)
            domain.define_function(Function(lower_case, FunctionTyping[BOOL](TYPE, OBJECT)))

        return domain

    def _get_all_obj_prop_vals(self, env, obj_props):
        all_objs = env.objs
        boolified_props = {}
        for _, instances in all_objs.items():
            for instance in instances:
                for prop in obj_props:
                    if hasattr(instance, prop):
                        val = getattr(instance, prop)
                        boolified_props.update({val:val})
                        # store in descriptor_store
                        prop = prop.capitalize()
                        if prop in self.descriptor_store:
                            if "descriptions" in self.descriptor_store[prop]:
                                if not val in self.descriptor_store[prop]["descriptions"]:
                                    self.descriptor_store[prop]["descriptions"].append(str(val))
                            else:
                                self.descriptor_store[prop]["descriptions"] = [str(val)]
                        else:
                            self.descriptor_store[prop] = {"descriptions":[], "values": None}
                            self.descriptor_store[prop]["descriptions"].append(str(val))
        return boolified_props

    def get_reward_n_done(self):
        return self.current_reward, self.task_done
    
    def get_mission(self):
        return self.mission




if __name__ == "__main__":
    env = CleaningUpTheKitchenOnlyEnv()
    sg = Scenegraph(env)
    sg.update()
    domain = sg.get_domain()
    domain.print_summary()
    
    sg.render(fancy_vis=True)

    # Interactive loop to accept arguments and call get_attr_for_id
    while True:
        user_input = input("Enter arguments for get_attr_for_id (attribute id [id2] [id3]) or 'q' to quit: ")
        if user_input.strip().lower() == 'q':
            break
        args = user_input.split()
        if len(args) < 2:
            print("Invalid input. Please enter at least an attribute and one id.")
            continue
        attribute = args[0]
        id = args[1]
        id2 = args[2] if len(args) > 2 else None
        id3 = args[3] if len(args) > 3 else None
        
        result = sg.get_attr_for_id(attribute, id, id2, id3)
        print("Result:", result)
    
