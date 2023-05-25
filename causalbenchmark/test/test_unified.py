from ..graphs import create_graph
from ..queries import create_query


# def test_generate_meta():
# 	example = {
# 		'phenomenon': 'confounding',
# 		'graph': 'V1->X,V1->Y,X->Y',
# 		'nodes': ['V1', 'X', 'Y'],
# 		'structuralEqs_type': 'bernoulli',
# 		'structuralEqs': {'p(V1)': 0.7,
# 		                  'p(X | V1)': [0.34, 0.025714285714285714],
# 		                  'p(Y | V1, X)': [[0.15, 0.55], [0.25, 0.65]]},
# 		'simpson': True,
# 		'query': {'query_type': 'ate',
# 		          'rung': 2,
# 		          'formal_form': 'E[Y|do(X = 1)] − E[Y|do(X = 0)]',
# 		          'candidate_adjustment_set': None,
# 		          'adjustment_treatment': 'X',
# 		          'adjustment_effect': 'Y'},
# 		'groundtruth': 0.3999999999999998,
# 		'given_info': [{'p(Y|V1,X)': [[0.15, 0.55], [0.25, 0.65]]}, {'p(V1)': 0.7}]
# 	}
#
#
# 	g = create_graph('confounding')
#
# 	q = create_query('ate')
#
# 	q_meta = q.meta_data()
#
# 	g_meta = g.meta_data()
#
# 	assert q_meta == example['query']


def test_entry():
	entry = {'ID': 0,
	         'descriptive_id': 'confounding_c_ate_smoke_birthWeight',
	         'sensical': 1,
	         'story_id': 'smoke_birthWeight',
	 'background': 'Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Low infant birth weight has a direct effect on smoking mother and infant mortality. Smoking mother has a direct effect on infant mortality.',
	 'given_info': 'For infants with normal birth weight and with nonsmoking mothers, the probability of infant mortality is 35.00%. '
	               'For infants with normal birth weight and with smoking mothers, the probability of infant mortality is 71.00%. '
	               'For infants with low birth weight and with nonsmoking mothers, the probability of infant mortality is 41.00%. '
	               'For infants with low birth weight and with smoking mothers, the probability of infant mortality is 85.00%. '
	               'The overall probability of low infant birth weight is 60.00%.',
	 'question': 'Will smoking mother increase the chance of infant mortality?',
	 'variable_mapping': {'V1name': "infant's birth weight", 'V10': 'normal infant birth weight',
	                      'V11': 'low infant birth weight', 'Xname': 'maternal smoking status',
	                      'X0': 'nonsmoking mother', 'X1': 'smoking mother', 'Yname': 'infant mortality',
	                      'Y0': 'absence of infant mortality', 'Y1': 'infant mortality'}, 'answer': 'yes',
	 'reasoning': {'step0': 'E[Y|do(X = 1)] − E[Y|do(X = 0)]',
	               'step1': 'P(Y=1|do(X=1)) - P(Y=1|do(X=0)) ',
	               'step2': '\\sum_{V1=v} P(V1=v)*[P(Y=1|V1=v,X=1) - P(Y=1|V1=v, X=0)]\n =P(V1=0)*[P(Y=1|V1=0,X=1)-P(Y=1|V1=0,X=0)]+P(V1=1)*[P(Y=1|V1=1,X=1)-P(Y=1|V1=1,X=0)]',
	               'step3': '0.4*[0.71-0.35]+0.6*[0.85-0.41]',
	               'step4': 0.408},
	 'meta': {'phenomenon': 'confounding', 'graph': 'V1->X,V1->Y,X->Y', 'nodes': ['V1', 'X', 'Y'],
	          'structuralEqs_type': 'bernoulli',
	          'structuralEqs': {'p(V1)': 0.6, 'p(X | V1)': [0.7391304347826083, 0.30434782608695626],
	                            'p(Y | V1, X)': [[0.35, 0.71], [0.41, 0.85]]}, 'simpson': True,
	          'query': {'query_type': 'ate', 'rung': 2, 'formal_form': 'E[Y|do(X = 1)] − E[Y|do(X = 0)]',
	                    'candidate_adjustment_set': None, 'adjustment_treatment': 'X', 'adjustment_effect': 'Y'},
	          'groundtruth': 0.40799999999999986,
	          'given_info': [{'p(Y|V1,X)': [[0.35, 0.71], [0.41, 0.85]]}, {'p(V1)': 0.6}]}}


	pass

















