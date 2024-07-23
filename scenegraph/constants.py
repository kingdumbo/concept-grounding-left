AGENT_RELATIVE_STATES = {
    "infovofrobot": "infront",
    "inhandofrobot": "inhand",
    "inreachofrobot": "inreach",
    "insameroomasrobot": "insameroom",
    "near": "near" # is evaluated explicitly in code
}

ABSOLUTE_STATES = {
    "cookable": "cookable",
    "dustyable": "dustable",
    "freezable": "freezable",
    "openable": "openable",
    "sliceable": "sliceable",
    "soakable": "soakable", 
    "stainable": "stainable", 
    "toggleable": "toggleable", 
    "onfloor": "onfloor", 
    "coldSource": "coldsource", 
    "cleaningTool": "cleaningtool"
}

RELATIONS = {
    "atsamelocation": "atsamelocation",
    "inside": "in",
    "nextto": "nextto",
    "onTop": "ontop",
    "under": "under"
}

OBJ_BOOL_PROPS = {
    "is_furniture": "furniture"
}

OBJ_PROPS = {
    "type":"type",
    "color": "color",
    "name": "name"
}#'actions', 'all_pos', 'can_contain', 'can_overlap', 'can_seebehind', 'check_abs_state', 'check_rel_state', 'color', 'contains', 'cur_pos', 'decode', 'dims', 'encode', 'get_ability_values', 'get_all_state_values', 'height', 'icon', 'init_pos', 'inside_of', 'is_furniture', 'load', 'name', 'possible_action', 'render', 'render_background', 'render_state', 'reset', 'states', 'type', 'update', 'update_pos', 'width'
