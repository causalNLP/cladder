# Causal Benchmark

This repo stores the codes for the Causal Benchmark project.

## Setup

To use the codes in this repo, first clone this repo:
    
    git clone https://github.com/causalNLP/causalbenchmark
    cd causalbenchmark

Then, install the dependencies:
    
    pip install -r requirements.txt

Finally, install the package:

    pip install -e .

Check to make sure everything is setup correctly by running the unit tests:

    pytest


## Update: 21 Apr

Generate demo data using

    fig generate demo

Checkout the corresponding config file [here](configs/demo.yaml).

And the script which is implemented in [generator.py](causalbenchmark/generator.py) - the function `generate_and_store`.

Also, you can run the unit tests with

    pytest


High level outline of todo (incomplete):

- [] Complete missing queries: counterfactuals, NDE, NIE, adjustment set, etc.
- [] Improve templates
- [] Filter out existing nonsense combinations
- [] Clean up meta data
- [] Replace MC integration with symbolic causal inference engine (Dowhy + CVXPY)


## Previous Notes

### Overview

#### Update

Current example questions can be viewed in `data/` directory.

To generate the data, run:

    fig generate prop

#### File Pointers

- (Private for members only) **All project files:** Shared [Google Drive folder](https://drive.google.com/drive/folders/1vis9JXdLhO5cD3rPsY7ZoYNrBZ-vzqHF) for the project
- **Up-to-date doc** to log the progress our data generation: [this doc](https://docs.google.com/document/d/1CTMIp5xy1Jh8P2eQFo4devYNwpXhVBqlSAoerQdVcgc/edit#heading=h.bvolnrpz0nl7)
- **Codes:** This repo, plus the [outputs](https://drive.google.com/drive/folders/1ZgmuKILg-B-xaQWlyczFWTcTQ74kKJG0) 
  folder in our [equivalent Google Drive folder](https://drive.google.com/drive/folders/1vWGXb1SCV8SygU98veS0sfHiZV-4o1ug). 
  Previously, we've also used [colabs](https://drive.google.com/drive/folders/1Sv_5JSN4My23KZAKJm-L2WZnyA6z5Ao_).
- **Data input:** Our [spreadsheet](https://docs.google.com/spreadsheets/d/1r-pBePcNmCopjBIe6lyuNV0vO7hFSwJS4Efn5n_W2Ps/edit#gid=0)



**Note:** 

- For project progress (which will change frequently, e.g., recent todos): feel free to directly update [our doc](https://docs.google.com/document/d/1CTMIp5xy1Jh8P2eQFo4devYNwpXhVBqlSAoerQdVcgc/edit#heading=h.bvolnrpz0nl7)

- For static documentation of the codes, feel free to just update this README.

### Structure of this Code Repo

Following are different parts of our codes:
1. **Dataset Composer:** [main_struc.py](main_struc.py)

   a. **Sampler:** Sampling each element based on dependencies

   b. **Formal Solver:** Generating the ground truth by a formal solver

   c. **Verbalizer:** Using text templates to generate natural language

2. **Inquirer:** Getting the responses from LLMs [get_llm_responses.py](get_llm_responses.py)

3. **Scorer:** Scoring how close the model responses is to the ground-truth answer


