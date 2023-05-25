
# CLadder: Evaluating Causal Reasoning in Language Models


## 🚀 Get the dataset now! [cladder-v1.zip](https://github.com/causalNLP/cladder/raw/main/data/cladder-v1.zip)

- zip file size: 6.5MB
- Version: v1
- Date: 2023-05-25


This repo contains the full CLadder dataset (and code) for evaluating (formal) causal reasoning in language models. The dataset asks yes/no questions in natural language that generally require statistical and causal inference to answer.



Although there are several different variants, the main dataset (including questions from all variants) is `cladder-v1-aggregate.json`, so that is the recommended file to use for most purposes.


Developed by: Zhijing Jin, Yuen Chen, Felix Leeb, Luigi Gresele, Ojasv Kamal, Zhiheng LYU, Kevin Blin, Fernando Gonzalez Adauto, Max Kleiman-Weiner, Mrinmaya Sachan, Bernhard Schölkopf



## Dataset

### Data Usage

The data is stored in JSON format, with each element representing a single question. Each question has the following fields:

- `question_id`: a unique (for the file) identifier for the question
- `desc_id`: a more descriptive identifier for the question (generally not needed)
- `given_info`: natural language supplementary information that should be given to the model to answer the question.
- `question`: the question itself, in natural language
- `answer`: the answer to the question {yes, no}
- `reasoning`: a step-by-step explanation of the causal reasoning used to answer the question
- `meta`: metadata about the question, including the following fields:
  - `query_type`: the type of question, one of {ATE, marginal, correlation, ETT, NDE, NIE, etc.}
  - `rung`: the rung of the ladder of causation that the question corresponds to
  - `story_id`: the id of the story used to verbalize the question
  - `graph_id`: the id of the causal graph structure used to verbalize the question
  - `model_id`: the id of the underlying model used to generate the question (corresponding to a model in `cladder-v1-meta-models.json`)
  - `groundtruth`: the groundtruth value of what the question is asking about

#### Prompting the Model

When evaluating a language model, it is recommended that the prompt includes 3 components:

1. The `background` field of the model corresponding to the question (found in `cladder-v1-meta-models.json` using the `model_id` field of the question's metadata).
2. The `given_info` field of the question.
3. The `question` field of the question.


#### Example

For example, the prompt corresponding to question 16825 (which asks about the average treatment effect for a simple instrumental variable setting) in `cladder-v1-aggregate.json` could be:


> Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Unobserved confounders has a direct effect on education level and salary. Proximity to a college has a direct effect on education level. Education level has a direct effect on salary. Unobserved confounders is unobserved.
>
> For people living far from a college, the probability of high salary is 35%. For people living close to a college, the probability of high salary is 53%. For people living far from a college, the probability of college degree or higher is 40%. For people living close to a college, the probability of college degree or higher is 73%.
>
> Will college degree or higher decrease the chance of high salary?

The associated reasoning steps found in the `reasoning` field are:
    
> Step 0: Let V2 = proximity to a college; V1 = unobserved confounders; X = education level; Y = salary. 
> 
> Step 1: V1->X,V2->X,V1->Y,X->Y 
> 
> Step 2: E[Y | do(X = 1)] - E[Y | do(X = 0)]
> 
> Step 3: [P(Y=1|V2=1)-P(Y=1|V2=0)]/[P(X=1|V2=1)-P(X=1|V2=0)]
> 
> Step 4: P(Y=1 | V2=0) = 0.35; P(Y=1 | V2=1) = 0.53; P(X=1 | V2=0) = 0.40; P(X=1 | V2=1) = 0.73
> 
> Step 5: (0.53 - 0.35) / (0.73 - 0.40) = 0.55
> 
> Solution: 0.55 > 0


Note that in addition to the `background` field, the model information found in `cladder-v1-meta-models.json` contains sufficient information to fully reconstruct the underlying causal model used to generate this question (and 59 others).

### Dataset Statistics

Here are some basic statistics for the main dataset (`cladder-v1-aggregate.json`).

Number of questions: 10,560
Answers: {"yes": 5,280, "no": 5,280}

Query Types:

| Query Type                             | Rung | Code               | Number  | Percent |
|----------------------------------------|------|--------------------|---------|---------|
| Correlation                            | 1    | correlation        | 1476    | 14.0%   |
| Marginal Distribution                  | 1    | marginal           | 1644    | 15.6%   |
| Expaining Away Effect                  | 1    | exp_away           | 168     | 1.6%    |
| Average Treatment Effect               | 2    | ate                | 1476    | 14.0%   |
| Backdoor Adjustment Set                | 2    | backadj            | 1644    | 15.6%   |
| Collider Bias                          | 2    | collider_bias      | 168     | 1.6%    |
| Effect of the Treatment on the Treated | 3    | ett                | 1296    | 12.3%   |
| Natural Direct Effect                  | 3    | nde                | 384     | 3.6%    |
| Natural Indirect Effect                | 3    | nie                | 828     | 7.8%    |
| Counterfactual (deterministic)         | 3    | det-counterfactual | 1476    | 14.0%   |


Graph Types:

| Graph Type   | Number  | Percent |
|--------------|---------|---------|
| IV           | 900     | 8.5%    |
| arrowhead    | 1536    | 14.5%   |
| chain        | 1092    | 10.3%   |
| collision    | 672     | 6.4%    |
| confounding  | 1008    | 9.5%    |
| diamond      | 1092    | 10.3%   |
| diamondcut   | 792     | 7.5%    |
| fork         | 1008    | 9.5%    |
| frontdoor    | 924     | 8.8%    |
| mediation    | 1536    | 14.5%   |



### Data Variants

If you want to dig a little deeper into understanding how well language models perform causal reasoning, we also include a few variants of the dataset (each of which contains about 10k questions, and the aggregate dataset is made up of an even mix of these variants):

- `cladder-v1-q-easy.json`: questions that are easy to answer (i.e. the causal mechanisms generally conform to what you would expect)
- `cladder-v1-q-hard.json`: the structure of the causal graph remains unchanged, but the strengths of causal mechanisms are generally counterintuitive
- `cladder-v1-q-commonsense.json`: an even mix of easy and hard questions
- `cladder-v1-q-anticommonsense.json`: for each causal graph we replace one of the variables (either treatment or outcome) with a randomly selected one that common sense would tell you is not related to the other variable at all.
- `cladder-v1-q-nonsense.json`: here the graph structure remains unchanged, but all variables are replaced from semantically meaningful concepts to randomly generated 4-letter words.


## Code Setup


To use the codes in this repo, first clone this repo:
    
    git clone https://github.com/causalNLP/causalbenchmark
    cd causalbenchmark

Then, install the dependencies:
    
    pip install -r requirements.txt

Finally, install the package:

    pip install -e .

Check to make sure everything is setup correctly by running the unit tests:

    pytest


## Code Usage

Generate demo data using

    fig generate demo

Checkout the corresponding config file [here](configs/demo.yaml).

And the script which is implemented in [generator.py](causalbenchmark/generator.py) - the function `generate_and_store`.

Also, you can run the unit tests with

    pytest

[more info coming soon]
