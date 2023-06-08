import numpy as np

from .. import util

from ..graphs import create_graph
from ..queries import create_query
from ..verbal import load_story
from ..graphs.builders import MechanismCorrelationBuilder, NiceSCMBuilder

from ._test_util import example_labels



def test_collider_bias():
	query = create_query('collider_bias',
	                     ask_polarities=True,
	                     ask_all_colliders=True,
	                     ask_treatments=True,
	                     ask_outcomes=True,
	                     ask_baselines=True,
	                     )

	scm = create_graph('collision', seed=101)
	labels = example_labels

	entries = [q for q in query.generate_questions(scm, labels)]

	print()
	print(len(entries))



def test_exp_away():
	query = create_query('exp_away',
	                     ask_polarities=True,
	                     ask_all_colliders=True,
	                     ask_treatments=True,
	                     ask_outcomes=True,
	                     ask_baselines=True,
	                     )

	scm = create_graph('collision', seed=17)
	labels = example_labels

	entries = [q for q in query.generate_questions(scm, labels)]

	print()
	print(len(entries))



def test_ate():
	query = create_query('ate',
	                     ask_treatments=True,
	                     ask_polarities=True,
	                     ask_outcomes=True,
	                     )

	# story =
	# scm = create_graph(story['graph'])
	# labels = story['labels']

	scm = create_graph('confounding', seed=10)
	labels = example_labels

	entries = [q for q in query.generate_questions(scm, labels)]

	print()
	print(len(entries))



# def test_ate_spec():
# 	query = create_query('ate',
# 	                     ask_treatments=False,
# 	                     ask_polarities=False,
# 	                     ask_outcomes=False,
# 	                     )
#
# 	graph_id = 'confounding'
#
# 	builder = MechanismCorrelationBuilder(seed=100)
# 	specs = list(builder.spawn_specs(graph_id))
# 	builder._rng.shuffle(specs)
# 	specs = specs[:10]
#
# 	labels = example_labels
#
# 	entries = [q for spec in specs
# 	           for q in query.generate_questions(builder.generate_constrained_scm(graph_id, spec), labels)]
#
# 	print()
# 	print(len(entries))



def test_ett():
	story = load_story('simpson_hospital')

	graph_id = story['phenomenon']
	labels = story

	# graph_id = 'confounding'
	# labels = example_labels

	query = create_query('ett',
	                     ask_treatments=True,
	                     ask_polarities=True,
	                     ask_outcomes=True,
	                     )

	# story =
	# scm = create_graph(story['graph'])
	# labels = story['labels']

	scm = create_graph(graph_id, seed=10)

	entries = [q for q in query.generate_questions(scm, labels)]

	print()
	print(len(entries))



# def test_ett_spec():
# 	story = load_story('simpson_drug')
#
# 	graph_id = story['phenomenon']
# 	labels = story
#
# 	query = create_query('ett',
# 	                     ask_treatments=False,
# 	                     ask_polarities=False,
# 	                     ask_outcomes=False,
# 	                     )
#
# 	# graph_id = 'confounding'
# 	# labels = example_labels
#
# 	builder = MechanismCorrelationBuilder(seed=100)
# 	specs = list(builder.spawn_specs(graph_id))
# 	builder._rng.shuffle(specs)
# 	specs = specs[:10]
#
# 	entries = [q for spec in specs
# 	           for q in query.generate_questions(builder.generate_constrained_scm(graph_id, spec), labels)]
#
# 	print()
# 	print(len(entries))



def test_nde():
	query = create_query('nde',
	                     ask_polarities=True,
	                     )

	scm = create_graph('mediation', seed=10)
	labels = example_labels

	entries = [q for q in query.generate_questions(scm, labels)]

	print()
	print(len(entries))


def test_nie():
	query = create_query('nie',
	                     ask_polarities=True,
	                     )

	scm = create_graph('mediation', seed=10)
	labels = example_labels

	entries = [q for q in query.generate_questions(scm, labels)]

	print()
	print(len(entries))


def test_marginals():
	query = create_query('marginal',
	                     ask_treatments=True,
	                     ask_polarities=True,
	                     )

	scm = create_graph('confounding', seed=10)
	labels = example_labels

	entries = [q for q in query.generate_questions(scm, labels)]

	print()
	print(len(entries))


def test_correlations():
	query = create_query('correlation',
	                     ask_treatments=True,
	                     ask_polarities=True,
	                     ask_results=True,
	                     )

	scm = create_graph('confounding', seed=10)
	labels = example_labels

	entries = [q for q in query.generate_questions(scm, labels)]

	print()
	print(len(entries))


def test_backadj():
	query = create_query('backadj',
	                     ask_flipped=True,
	                     ask_polarities=True,
	                     ask_all_adj=True,
	                     ask_all_bad=True,
	                     )

	scm = create_graph('frontdoor', seed=10)
	labels = example_labels

	entries = [q for q in query.generate_questions(scm, labels)]

	print()
	print(len(entries))



def test_deterministic_counterfactual():

	query = create_query('det-counterfactual')

	scm = create_graph('det-mediation')

	questions = list(query.generate_questions(scm, example_labels))

	assert len(questions) == 4

	# query2 = create_query('det-counterfactual', treatment=None)
	#
	# questions = list(query2.generate_questions(scm, example_labels))
	#
	# assert len(questions) == 8



# def test_deterministic_interventions():
#
# 	query = create_query('interv_y')
#
# 	scm = create_graph('det-twocauses')
#
# 	questions = list(query.generate_questions(scm, example_labels))
# 	print(questions)
#
# 	assert len(questions) == 8
#
# 	query2 = create_query('interv_y', treatment=None)
#
# 	questions = list(query2.generate_questions(scm, example_labels))
#
# 	assert len(questions) == 16



def test_deterministic_counterfactual2():

	query = create_query('det-counterfactual')

	scm = create_graph('det-arrowhead')

	questions = list(query.generate_questions(scm, example_labels))

	assert len(questions) == 8






# def test_ate():
#
# 	query = create_query('ate')
#
# 	scm = create_graph('example2')
#
# 	questions = list(query.generate_questions(scm, example_labels))
#
# 	ate = questions[0]['meta']['ate']
#
# 	assert np.isclose(ate, scm.Y.param[1] - scm.Y.param[0])
#
# 	assert len(questions) == 8
#
# 	assert query._sol_verbalization[ate > 0] == questions[-1]['answer']
#
#
#
# def test_ett():
#
# 	query = create_query('ett')
#
# 	scm = create_graph('example2', seed=100)
#
# 	questions = list(query.generate_questions(scm, example_labels))
#
# 	assert len(questions) == 8
#
# 	ett = questions[-1]['meta']['ett']
#
# 	assert np.isclose(ett, scm.Y.param[1] - scm.Y.param[0])




















