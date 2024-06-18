import sys
sys.path.append("/home/max/uni/LEFT/Jacinle")
sys.path.append("/home/max/uni/LEFT/")
sys.path.append("/home/max/uni/LEFT/scenegraph/")

from constants import AGENT_RELATIVE_STATES, ABSOLUTE_STATES, RELATIONS
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
        #JUST TRYING
        self.rendering = False

        self.env = env

        # set types of states
        self.agent_rel_states = AGENT_RELATIVE_STATES
        self.abs_states = ABSOLUTE_STATES
        self.attributes = [*self.agent_rel_states, *self.abs_states]
        # TODO include type in attributes (furniture)
        self.relations = RELATIONS

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

    def update(self):
        state = self.env.get_state()
        self.reset() # always start from empty graph

        # add node for agent
        self.sg.add_node("Agent")
        self.sg_color_map.append("red")
        self.obj_ids.append("Agent")

        # TODO add agent position and orientation
       
        # FIRST:
        # add node for each object together with absolute properties
        # and agent-relative states
        
        # get objects (including furniture and other objects)
        objs = state["objs"]
    
        # iterate over all objects
        for label, obj_list in objs.items():
            for idx, obj in enumerate(obj_list):
                
                # label and index jointly unique
                id = f"{label}_{str(idx)}"
    
                # add for lookup
                self.obj_to_node[obj] = id
                self.node_to_obj[id] = obj
                self.obj_ids.append(id)
    
                # extract absolute and agent-relative attributes
                obj_attributes = {}
                obj_agent_relations = {}
    
                # check if furniture, if yes, connect with agent
                # and color accordingly
                obj_attributes["type"] = "object" 
                color = "blue"
                if obj.is_furniture():
                    obj_attributes["type"] =  "furniture"

                    # add edge between furniture and agent
                    self.sg.add_edge(id, "Agent", isnear=True)
                    color = "green"

                    # add to value store 
                    self.set_attr_for_id("isnear", 1, id)

                self.sg_color_map.append(color)
    
    
                for attribute, state in obj.states.items():
                    # check if is agent-relative
                    # NOTE: agent relative attributes are treated as essentially absolute attributes of the object
                    if attribute in self.agent_rel_states:
                        val = state.get_value(self.env)
                        if val is True:
                            # add for graph
                            obj_agent_relations[attribute] = val
                            
                            # add to store
                            self.set_attr_for_id(attribute, 1, id)
    
                    # otherwise check if absolute attribute
                    elif attribute in self.abs_states:
                        val = state.get_value(self.env)
                        if val is True:
                            # add for graph
                            obj_attributes[attribute] = val

                            # add to store
                            self.set_attr_for_id(attribute, 1, id)
    
                    # for completeness check if relative, but will be implemented later 
                    elif attribute in self.relations:
                        pass
    
                    else:
                        raise NotImplementedError(f"Unknown attribute: {attribute}")
                    
                # add node with node attributes
                self.sg.add_node(id, **obj_attributes)
    
                # add edge to agent if at least one attribute
                if len(obj_agent_relations.keys()) > 0:
                    self.sg.add_edge(id, "Agent", **obj_agent_relations)
    
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
                for relation in self.relations:
                    try:
                        val = obj1.states[relation].get_value(obj2, self.env)
                        if val is True:
                            # add for graph
                            relations_obj1_obj2[relation] = val

                            # add to store
                            obj_id_from = self.obj_to_node[obj1]
                            obj_id_to = self.obj_to_node[obj2]
                            self.set_attr_for_id(relation, 1, obj_id_from, obj_id_to)
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
        num_atts= len(self.attributes)
        num_rels = len(self.relations)
        self.attribute_vals = torch.zeros((num_atts, num_env_objects))
        self.relation_vals = torch.zeros((num_rels, num_env_objects, num_env_objects))

        # reset ordering of ids
        self.obj_ids =  []

        # reset lookups
        self.obj_to_node = {}
        self.node_to_obj = {}


    def id_to_idx(self, id):
        if id in self.obj_ids:
            return self.obj_ids.index(id)
        raise ValueError(f"Unkown node-id: {id}")


    def attr_to_idx(self, category, name):
        # id of attribute to index on vector of all feature vectors
        # get correct list of attributes to index from
        if not category in ["attribute", "relation"]: raise NotImplementedError("Multi-relation not implemented on graph")
        attribute_list = getattr(self, f"{category}s")

        # get index if the name of the attribute as index
        if not name in attribute_list: raise ValueError(f"{name} not in list of possible {category}s")

        return attribute_list.index(name)


    def set_attr_for_id(self, attribute, value, id, id2=None, id3=None):
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

        elif id2 is not None and id3 is not None:
            raise NotImplementedError("Multi-relation not implemented on graph")

        else:
            # if none applies, somethings wrong
            raise LookupError(f"No category for id={id}, id2={id2}, id3={id3}")
        
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
                net.write_html(self.html_filename, open_browser =False)
                
                if not self.rendering:
                    import threading
                    self.rendering=True
                    server = Server()
                    path = "/home/max/uni/LEFT/network.html"
                    server.watch(path)
                    server_thread = threading.Thread(target=server.serve)
                    server_thread.setDaemon(True)
                    server_thread.start()


    def get_domain(self):
        # constructs domain from graph/evnv
        domain = create_bare_domain()

        OBJECT = ObjectType("Object")
        
        # create functions for the attributes
        for attr in self.attributes:
            #domain.define_function(Function(f"{attr}_Object", FunctionTyping[BOOL](OBJECT)))
            domain.define_function(Function(attr, FunctionTyping[BOOL](OBJECT)))

        # create functions for the relations
        for rel in self.relations:
            domain.define_function(Function(rel, FunctionTyping[BOOL](OBJECT,OBJECT)))

        return domain



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
    
