import numpy as np
from .. import util
from functools import lru_cache

from .base import MetricQueryType



class ManualEstimandQueryType(MetricQueryType):
	'''A query type that requires the user to manually estimate the estimand.'''
	name = None
	rung = None
	formal_form = None

	_known_estimand_terms = {}
	_known_estimands = {}
	_known_calculations = {}


	@lru_cache(maxsize=None)
	def _check_known_estimand_terms(self, scm):
		if self.treatment != 'X' or self.outcome != 'Y':
			raise NotImplementedError('Only X->Y is supported')

		structure = scm.symbolic_graph_structure()
		terms = self._known_estimand_terms.get(structure, None)
		if terms is None:
			raise self._QueryFailedError(f'No estimand found for graph: {structure}')
		return list(map(util.parse_mechanism_str, terms))


	@lru_cache(maxsize=None)
	def _check_known_estimands(self, scm):
		if self.treatment != 'X' or self.outcome != 'Y':
			raise NotImplementedError('Only X->Y is supported')

		structure = scm.symbolic_graph_structure()
		estimands = self._known_estimands.get(structure, None)
		if estimands is None:
			raise self._QueryFailedError(f'No estimand found for graph: {structure}')
		return estimands


	def verbalize_given_info(self, scm, labels, **details):
		terms = self._check_known_estimand_terms(scm)
		mechs = [scm.distribution(name, *parents) for name, parents in terms]
		# info = scm.verbalize_mechanism_details(labels, mechs)
		info = scm.verbalize_description(labels, mechanisms=mechs)
		return info


	def symbolic_given_info(self, scm, **details):
		terms = self._check_known_estimand_terms(scm)
		mechs = [scm.distribution(name, *parents) for name, parents in terms]
		return {str(mech): mech.param.tolist() for mech in mechs}


	def symbolic_estimand(self, scm, **details):
		estimand = self._check_known_estimands(scm)[0]
		return estimand


	def meta_data(self, scm, labels):
		return {
			'query_type': self.name,
	        'rung': self.rung,
	        'formal_form': self.formal_form.format(treatment=self.treatment, outcome=self.outcome),

			'given_info': self.symbolic_given_info(scm),
			'estimand': self.symbolic_estimand(scm),

			'treatment': self.treatment,
			'outcome': self.outcome,
		}


	def reasoning(self, scm, labels, entry):
		steps = {}

		steps['step0'] = 'Let ' + '; '.join(util.pformat('{v.name} = {{v.name}name}', v=v, **labels)
		                                  for v in scm.variables()) + '.'

		steps['step1'] = scm.symbolic_graph_structure()


		steps['step2'] = entry.get('meta', {}).get('formal_form', '')

		steps['step3'] = entry.get('meta', {}).get('estimand', '')

		given = entry.get('meta', {}).get('given_info', None)

		terms = util.parse_given_info(given)
		steps['step4'] = '\n'.join(terms)


		calc = self._known_calculations.get(steps['step1'], '')

		gt = entry.get('meta', {}).get('groundtruth', None)

		steps['step5'] = util.pformat(calc, given=given, result=gt, **labels)

		if abs(gt) < 0.005:
			steps['end'] = '0.00 = 0'
		else:
			steps['end'] = f'{gt:.2f} > 0' if gt > 0 else f'{gt:.2f} < 0'

		return steps













