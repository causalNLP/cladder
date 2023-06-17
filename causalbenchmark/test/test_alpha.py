# from ..graphs.revised import GraphProperty,Estimand, LinearGraph,PropGraph,BernGraph
from .. import util
# from causalnlp import ci_relation_check
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# from  ..verbal import Verbalizer

from ..graphs import create_graph
from ._test_util import example_labels

# phenomenon_list = ['confounding', 'mediation', 'chain', 'arrowheadcollision', 'collision', 'arrowheadmediation',
#                    'det-diamond', 'det-twocauses', 'det-triangle', 'IV', 'frontdoor']
# def test_question():
#     all_phennmenon = {}
#     for p in phenomenon_list:
#         asker = Verbalizer(p)
#         all_phennmenon[p] = asker.default_all_questions()



def test_background():
    g = create_graph('IV', seed=10)

    print()

    bg = g.verbalize_background(example_labels)

    goal_bg = 'Imagine a self-contained, hypothetical world with only the following conditions, ' \
              'and without any unmentioned factors or causal relationships: ' \
              'V1 has a direct effect on X and Y. ' \
              'V2 has a direct effect on X. ' \
              'X has a direct effect on Y. V1 is unobserved.'

    print(bg)
    assert bg == goal_bg



def test_description():
    g = create_graph('IV', seed=10)

    print()

    goal_desc = 'The overall probability of V2=1 is 43%. ' \
                'The overall probability of V1=1 is 96%. ' \
                'For V1=0 and V2=0, the probability of X=1 is 51%. ' \
                'For V1=0 and V2=1, the probability of X=1 is 14%. ' \
                'For V1=1 and V2=0, the probability of X=1 is 69%. ' \
                'For V1=1 and V2=1, the probability of X=1 is 84%. ' \
                'For V1=0 and X=0, the probability of Y=1 is 96%. ' \
                'For V1=0 and X=1, the probability of Y=1 is 21%. ' \
                'For V1=1 and X=0, the probability of Y=1 is 83%. ' \
                'For V1=1 and X=1, the probability of Y=1 is 15%.'

    desc = g.verbalize_description(example_labels)

    print(desc)
    assert desc == goal_desc










