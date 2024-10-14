# Grounding an LLM planner on a Scene Graph using the LEFT framework

A project work for Intelligent Robot Manipulation at TU Darmstadt SS '24.
The full report can be found [here](https://www.notion.so/elenamax/Grounding-an-LLM-planner-on-a-Scene-Graph-using-the-LEFT-framework-44ed3ccc9aad4f5fa6006f4449e5e600).



## Installation

Start by cloning the repo using:

```bash
  git clone https://github.com/kingdumbo/concept-grounding-left
```
Then navigate into the newly created folder. Set up your conda environment and install the dependencies:
```bash
  conda env create -f environment.yml
  conda activate concept-grounding
```
Then you need to install the dependencies *in order*:

### For LEFT
Put the Jacinle folder *at the root* of the project and add to the path:

Install [Jacinle](https://github.com/vacancy/Jacinle).
```bash
  git clone https://github.com/vacancy/Jacinle --recursive
  export PATH=<PATH_TO_JACINLE>/bin:$PATH
```

Install [Concepts](https://github.com/concepts-ai/concepts).
```bash
  git clone https://github.com/concepts-ai/Concepts.git
  cd Concepts
  pip install -e .
```

### Other dependencies
Some pip dependencies are necessary:
```bash
  pip install networx opencv-python pyvis livereload fuzzywuzzy PyYAML peewee
```

### Mini-BEHAVIOR
First, some dependencies:
```bash
pip install gym-minigrid==1.0.3
pip install setuptools==65.5.0 "wheel<0.40.0" 
pip install gym==0.21.0
```
And finally:
```bash
  cd mini_behavior
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

