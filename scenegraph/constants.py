AGENT_RELATIVE_STATES = {
    "infovofrobot": "infront",
    "inhandofrobot": "inhand",
    "inreachofrobot": "inreach",
    "insameroomasrobot": "insameroom",
    "near": "near" # is evaluated explicitly in code
}

ABSOLUTE_STATES = {
    "cookable": "cooked",
    "dustyable": "dusty",
    "freezable": "frozen",
    "openable": "open",
    "sliceable": "sliced",
    "soakable": "soaked", 
    "stainable": "stained", 
    "toggleable": "toggled", 
    "onfloor": "onfloor", 
    "coldSource": "coldsource", 
    "cleaningTool": "cleaningtool"
}

RELATIONS = {
    "atsamelocation": "atsamelocation",
    "inside": "inside",
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
}
