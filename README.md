
# Grounding an LLM planner on a Scene Graph using the LEFT framework

A project work for Intelligent Robot Manipulation at TU Darmstadt SS '24.
The full report can be found [here](https://www.notion.so/elenamax/Grounding-an-LLM-planner-on-a-Scene-Graph-using-the-LEFT-framework-44ed3ccc9aad4f5fa6006f4449e5e600).



## Installation

Start by cloning the repo using:

```bash
  git clone https://github.com/kingdumbo/concept-grounding-left --recursive
```
Then navigate into the newly created folder. Set up your conda environment and install the dependencies:
```bash
  conda env create -f environment.yaml
  conda activate left
```
Then you need to install local dependencies:

### Jacinle (for LEFT)
Add Jacinle to your path:
```bash
  export PATH=<PATH_TO_JACINLE>/bin:$PATH
```

### Concepts (for LEFT)
Navigate to the folder ./Concepts and install locally:
```bash
  pip install -e .
```

### Mini-BEHAVIOR
Navigate to the folder ./mini_behavior and install locally:
```bash
  pip install -e .
```

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`OPENAI_API_KEY`


## Run Locally

To re-run the experiments, you should be in the top-directory of the project and run with the environment activated:

### Scene Graph Interrogation
```bash
  python scenegraph/scenegraph_oracle.py
```

### Long-horizon Task Execution
```bash
    python scenegraph/llm_actor.py
```
## Authors

- [@Max Schindler](https://github.com/kingdumbo)

