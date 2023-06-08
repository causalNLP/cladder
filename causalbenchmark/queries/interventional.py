from typing import Generator
import numpy as np
from itertools import product
from .. import util
from .base import hparam, register_query, MetricQueryType
from .causal import ManualEstimandQueryType


@register_query('ate')
class AverageTreatmentEffectQuery(ManualEstimandQueryType):
	name = 'ate'
	rung = 2
	formal_form = 'E[{outcome} | do({treatment} = 1)] - E[{outcome} | do({treatment} = 0)]'

	_known_estimand_terms = {
		# chain
		"X->V2,V2->Y": ['P(Y|X)'],
		"X->Y,V2->Y": ['P(Y|X)'],
		# collision
		"X->V3,Y->V3": None,
		# confounding
		"V1->X,V1->Y,X->Y": ['P(Y|V1,X)', 'P(V1)'],
		# mediation
		"X->V2,X->Y,V2->Y": ['P(Y|X)'],
		# frontdoor
		"V1->X,X->V3,V1->Y,V3->Y": ['P(V3|X)', 'P(Y|X,V3)', 'P(X)'],
		# IV
		"V1->X,V2->X,V1->Y,X->Y": ['P(Y|V2)', 'P(X|V2)'],
		# arrowhead
		"X->V3,V2->V3,X->Y,V2->Y,V3->Y": ['P(Y|X)'],
		# nondet-diamond
		"X->V3,X->V2,V2->Y,V3->Y": ['P(Y|X)'],
		# nondet-diamondcut
		"V1->V3,V1->X,X->Y,V3->Y": ['P(Y|V1,X)', 'P(V1)'],
	}
	_known_estimands = {
		# chain
		"X->V2,V2->Y": ['P(Y=1|X=1) - P(Y=1|X=0)'],
		"X->Y,V2->Y": ['P(Y|X)'],
		# collision
		"X->V3,Y->V3": None,
		# confounding
		"V1->X,V1->Y,X->Y": ['\\sum_{V1=v} P(V1=v)*[P(Y=1|V1=v,X=1) - P(Y=1|V1=v, X=0)]'],
		# mediation
		"X->V2,X->Y,V2->Y": ['P(Y=1|X=1) - P(Y=1|X=0)'],
		# frontdoor
		"V1->X,X->V3,V1->Y,V3->Y": [
			'\\sum_{V3 = v} [P(V3 = v|X = 1) - P(V3 = v|X = 0)] * [\\sum_{X = h} P(Y = 1|X = h,V3 = v)*P(X = h)]'],
		# IV
		"V1->X,V2->X,V1->Y,X->Y": ['[P(Y=1|V2=1)-P(Y=1|V2=0)]/[P(X=1|V2=1)-P(X=1|V2=0)]'],
		# arrowhead
		"X->V3,V2->V3,X->Y,V2->Y,V3->Y": ['P(Y=1|X=1) - P(Y=1|X=0)'],
		# nondet-diamond
		"X->V3,X->V2,V2->Y,V3->Y": ['P(Y=1|X=1) - P(Y=1|X=0)'],
		# nondet-diamondcut
		"V1->V3,V1->X,X->Y,V3->Y": ['\\sum_{V1=v} P(V1=v)*[P(Y=1|V1=v,X=1) - P(Y=1|V1=v, X=0)]'],
	}

	_known_calculations = {

		"X->V2,V2->Y": '{given["p(Y | X)"][1]:.2f} - {given["p(Y | X)"][0]:.2f} = '
		               '{given["p(Y | X)"][1] - given["p(Y | X)"][0]:.2f}',
		'X->V2,X->Y,V2->Y': '{given["p(Y | X)"][1]:.2f} - {given["p(Y | X)"][0]:.2f} = '
		               '{given["p(Y | X)"][1] - given["p(Y | X)"][0]:.2f}',
		"X->Y,V2->Y": '{given["p(Y | X)"][1]:.2f} - {given["p(Y | X)"][0]:.2f} = '
		               '{given["p(Y | X)"][1] - given["p(Y | X)"][0]:.2f}',
		"X->V3,X->V2,V2->Y,V3->Y": '{given["p(Y | X)"][1]:.2f} - {given["p(Y | X)"][0]:.2f} = '
		               '{given["p(Y | X)"][1] - given["p(Y | X)"][0]:.2f}',
		"X->V3,V2->V3,X->Y,V2->Y,V3->Y": '{given["p(Y | X)"][1]:.2f} - {given["p(Y | X)"][0]:.2f} = '
		               '{given["p(Y | X)"][1] - given["p(Y | X)"][0]:.2f}',

		"V1->V3,V1->X,X->Y,V3->Y": '{1-given["p(V1)"][0]:.2f} '
		                           '* ({given["p(Y | V1, X)"][0][1]:.2f} - {given["p(Y | V1, X)"][0][0]:.2f}) '
		                           '{given["p(V1)"][0]:.2f} '
		                           '* ({given["p(Y | V1, X)"][1][1]:.2f} - {given["p(Y | V1, X)"][1][0]:.2f}) '
		                           '= {result:.2f}',

		"V1->X,V2->X,V1->Y,X->Y": '({given["p(Y | V2)"][1]:.2f} - {given["p(Y | V2)"][0]:.2f}) '
		                          '/ ({given["p(X | V2)"][1]:.2f} - {given["p(X | V2)"][0]:.2f}) = {result:.2f}',

		"V1->X,V1->Y,X->Y": '{1-given["p(V1)"][0]:.2f} '
                     '* ({given["p(Y | V1, X)"][0][1]:.2f} - {given["p(Y | V1, X)"][0][0]:.2f}) '
                     '+ {given["p(V1)"][0]:.2f} '
                     '* ({given["p(Y | V1, X)"][1][1]:.2f} - {given["p(Y | V1, X)"][1][0]:.2f}) '
                     '= {result:.2f}',

		"V1->X,X->V3,V1->Y,V3->Y": '({given["p(V3 | X)"][1]:.2f} - {given["p(V3 | X)"][0]:.2f})'
		                           '* ({given["p(Y | X, V3)"][1][1]:.2f} * {given["p(X)"][0]:.2f}) '
		                           '+ ({1-given["p(V3 | X)"][1]:.2f} - {1-given["p(V3 | X)"][0]:.2f}) '
		                           '* ({given["p(Y | X, V3)"][0][1]:.2f} * {1-given["p(X)"][0]:.2f}) '
		                           '= {result:.2f}',

	}


	question_template = hparam('Will {{treatment}{int(treated)}_noun} {"increase" if {polarity} else "decrease"} '
	                           'the chance of {{outcome}{int(result)}_noun}?')

	ask_treatments = hparam(True)
	ask_polarities = hparam(True)
	ask_outcomes = hparam(False)


	def generate_questions(self, scm, labels):

		# ate = E[Y | do(X=1)] - E[Y | do(X=0)]

		# do(X=1) increases the chance of Y=1 -> ate > 0
		# do(X=1) decreases the chance of Y=1 -> ate < 0
		# do(X=1) increases the chance of Y=0 -> ate < 0
		# do(X=1) decreases the chance of Y=0 -> ate > 0
		# do(X=0) increases the chance of Y=1 -> ate < 0
		# do(X=0) decreases the chance of Y=1 -> ate > 0
		# do(X=0) increases the chance of Y=0 -> ate > 0
		# do(X=0) decreases the chance of Y=0 -> ate < 0

		try:
			meta = self.meta_data(scm, labels)
			given_info = self.verbalize_given_info(scm, labels)
		except AttributeError as e:
			raise self._QueryFailedError(str(e))

		for treated in ([True, False] if self.ask_treatments else [True]):

			lb, ub = scm.ate_bounds(self.outcome, self.treatment, treated=treated)

			sign = None if lb*ub <= 0 else (1 if ub > 0 else -1)

			for result, polarity in product([True, False] if self.ask_outcomes else [True],
			                                [True, False] if self.ask_polarities else [True]):
				if sign is None:
					answer = 0
				elif abs(lb) < 0.005:
					answer = -1
				else:
					answer = sign * (-1) ** (1-int(result) + 1-int(polarity))

				yield {
					'given_info': given_info,
					'question': util.pformat(self.question_template,
					                         treatment=self.treatment,
					                         outcome=self.outcome,
					                         treated=treated,
					                         result=result,
					                         polarity=polarity,
					                         answer=answer,
					                         **labels),
					'answer': util.pformat(self.answer_template,
					                       treatment=self.treatment,
					                       outcome=self.outcome,
					                       treated=treated,
					                       result=result,
					                       polarity=polarity,
					                       answer=answer,
					                       **labels),

					'meta': {
						'treated': treated,
						'result': result,
						'polarity': polarity,
						'groundtruth': lb,
						**meta,
					}
				}





@register_query('backadj')
class BackdoorAdjustmentSetQuery(MetricQueryType):
	name = 'backadj'
	rung = 2
	formal_form = '[backdoor adjustment set for {outcome} given {treatment}]'

	correlation_template = hparam('how {{treatment}name} correlates with {{outcome}name}')
	repeat_template = 'this correlation'

	empty_template = hparam('We look directly at {correlation} in general')
	set_template = hparam('We look at {correlation} case by case according to {candidate_set}')
	set_element_template = hparam('{{term}name}')

	question_template = hparam('To understand how {{treatment}name} affects {{outcome}name}, '
	                           'is it more correct to use the Method {1 if {polarity} else 2} '
	                           'than Method {2 if {polarity} else 1}?')

	given_info_template = hparam('Method 1: {method1}. Method 2: {method2}.')

	ask_polarities = hparam(False)
	ask_flipped = hparam(True)
	ask_all_adj = hparam(False)
	ask_all_bad = hparam(False)

	_known_adjustment_sets = {
		# chain
		"X->V2,V2->Y": [[]],
		"X->Y,V2->Y": [[]],
		# collision
		"X->V3,Y->V3": [[]],
		# confounding
		"V1->X,V1->Y,X->Y": [['V1']],
		# mediation
		"X->V2,X->Y,V2->Y": [[]],
		# frontdoor
		"V1->X,X->V3,V1->Y,V3->Y": [['V1']],
		# IV
		"V1->X,V2->X,V1->Y,X->Y": [['V1'], ['V2', 'V1']],
		# arrowhead
		"X->V3,V2->V3,X->Y,V2->Y,V3->Y": [[], ['V2']],
		# nondet-diamond
		"X->V3,X->V2,V2->Y,V3->Y": [[]],
		# nondet-diamondcut
		"V1->V3,V1->X,X->Y,V3->Y": [['V3'], ['V1', 'V3'], ['V1']],
	}


	def reasoning(self, scm, labels, entry):
		return None # TODO


	def verbalize_given_info(self, scm, labels, candidate1, candidate2, **details):

		corr = util.pformat(self.correlation_template, treatment=self.treatment, outcome=self.outcome, **labels)
		repeat = util.pformat(self.repeat_template, treatment=self.treatment, outcome=self.outcome, **labels) \
			if self.repeat_template is not None and len(candidate2) else corr

		if len(candidate1):
			cand1 = util.verbalize_list(
				util.pformat(self.set_element_template, term=term, **labels) for term in candidate1)
			method1 = util.pformat(self.set_template, correlation=corr, candidate_set=cand1, **labels)
		else:
			method1 = util.pformat(self.empty_template, correlation=corr, **labels)

		if len(candidate2):
			cand2 = util.verbalize_list(
				util.pformat(self.set_element_template, term=term, **labels) for term in candidate2)
			method2 = util.pformat(self.set_template, correlation=repeat, candidate_set=cand2, **labels)
		else:
			method2 = util.pformat(self.empty_template, correlation=repeat, **labels)

		return util.pformat(self.given_info_template, method1=method1, method2=method2, **labels)


	def symbolic_given_info(self, scm, **details):

		if self.treatment != 'X' or self.outcome != 'Y':
			raise NotImplementedError('Only X->Y is supported')

		structure = scm.symbolic_graph_structure()
		terms = self._known_adjustment_sets.get(structure, None)
		if terms is None:
			raise self._QueryFailedError(f'No estimand found for graph: {structure}')

		return terms


	def _generate_wrong_adjustement_set(self, scm, all_adj):
		vars = [v for v in scm.variable_names() if v != self.treatment and v != self.outcome]

		for bad in util.powerset(vars):
			bad = list(bad)
			if bad not in all_adj:
				yield bad


	def generate_questions(self, scm, labels):

		try:
			meta = self.meta_data(scm, labels)
			# given_info = self.verbalize_given_info(scm, labels)
			# background = self.verbalize_background(scm, labels)
		except AttributeError as e:
			raise self._QueryFailedError(str(e))

		all_adj = list(self.symbolic_given_info(scm))
		all_bad = list(self._generate_wrong_adjustement_set(scm, all_adj))

		assert len(all_adj) > 0, 'No backdoor adjustment set found'
		assert len(all_bad) > 0, 'No invalid backdoor adjustment set found'

		if not self.ask_all_adj:
			all_adj = [all_adj[0]]
		if not self.ask_all_bad:
			all_bad = [all_bad[0]]

		for adj, bad in product(all_adj, all_bad):
			for flipped, polarity in product([True, False] if self.ask_flipped else [True],
			                                [True, False] if self.ask_polarities else [True]):

				answer = 1 if (polarity ^ flipped) else -1

				cand1, cand2 = (bad, adj) if flipped else (adj, bad)
				given_info = self.verbalize_given_info(scm, labels, candidate1=cand1, candidate2=cand2)

				yield {
					'given_info': given_info,
					'question': util.pformat(self.question_template,
				                        treatment=self.treatment,
				                        outcome=self.outcome,
				                        polarity=polarity,
				                         flipped=flipped,
				                        **labels),
					'answer': util.pformat(self.answer_template,
					                       treatment=self.treatment,
					                       outcome=self.outcome,
					                       flipped=flipped,
					                       polarity=polarity,
					                       answer=answer,
					                       **labels),

					# 'background': background,

					'meta': {
						'flipped': flipped,
						'polarity': polarity,
						'groundtruth': list(adj),
						'bad_candidate_set': list(bad),
						'given_info': [cand1, cand2],
						**meta,
					}
				}



	def meta_data(self, scm, labels):
		return {
			'query_type': self.name,
	        'rung': self.rung,
	        'formal_form': self.formal_form.format(treatment=self.treatment, outcome=self.outcome),

			'treatment': self.treatment,
			'outcome': self.outcome,
		}




# ask_treatments = hparam(True)
	# ask_polarities = hparam(True)
	# ask_outcomes = hparam(False)



@register_query('collider_bias')
class ColliderBiasQuery(MetricQueryType):
	name = 'collider_bias'
	rung = 2
	formal_form = 'E[{outcome} = {int(result)} | do({treatment} = {int(treated)}), {collider} = {int(baseline)}] ' \
	              '- E[{outcome} = {int(result)} | do({treatment} = {int(not treated)}), {collider} = {int(baseline)}]'


	given_info_template = hparam('For {{collider}{int(baseline)}_wheresentence}, the correlation between '
	                             '{{treatment}{int(treated)}_noun} and {{outcome}{int(result)}_noun} is '
	                             '{correlation:.2f}.')

	symbolic_template = 'P({outcome} = {int(result)} | {treatment} = {int(treated)}, {collider} = {int(baseline)}) ' \
						'- P({outcome} = {int(result)} | {treatment} = {int(not treated)}, {collider} = {int(baseline)})'


	question_template = hparam('If we look at {{collider}{int(baseline)}_wheresentence}, '
	                           'does it mean that {{treatment}{int(treated)}_noun} '
	                           '{"affects" if {polarity} else "does not affect"} '
	                           '{{outcome}{int(result)}_noun}?')

	ask_polarities = hparam(True)

	ask_all_colliders = hparam(False)

	ask_treatments = hparam(True)
	ask_outcomes = hparam(False)
	ask_baselines = hparam(False)


	def reasoning(self, scm, labels, entry):

		steps = {}

		steps['step0'] = 'Let ' + '; '.join(util.pformat('{v.name} = {{v.name}name}', v=v, **labels)
		                                  for v in scm.variables()) + '.'

		steps['step1'] = scm.symbolic_graph_structure()


		steps['step2'] = entry.get('meta', {}).get('formal_form', '')

		steps['step3'] = 'X and Y do not affect each other.'

		given = entry.get('meta', {}).get('given_info', None)

		# terms = util.parse_given_info(given)
		steps['step4'] = ''

		gt = entry.get('meta', {}).get('groundtruth', None)

		steps['step5'] = '0'# util.pformat(calc, given=given, result=gt, **labels)

		steps['end'] = gt# f'{gt:.2f} > 0' if gt > 0 else f'{gt:.2f} < 0'

		return steps


	def generate_questions(self, scm, labels):
		try:
			colliders = list(scm.find_colliders(self.treatment, self.outcome))
		except NotImplementedError:
			raise self._QueryFailedError('Collider detection not implemented for this SCM')
		if not len(colliders):
			raise self._QueryFailedError('No collider found')
		if not self.ask_all_colliders:
			colliders = [colliders[0]]

		try:
			meta = self.meta_data(scm, labels)
			# given_info = self.verbalize_given_info(scm, labels)
			# background = self.verbalize_background(scm, labels)
		except AttributeError as e:
			raise self._QueryFailedError(str(e))

		for collider in colliders:
			for baseline in [True, False] if self.ask_baselines else [True]:

				base_corr = scm.correlation(self.treatment, self.outcome, **{collider: int(baseline)})

				for treated, result, polarity in product(
						[True, False] if self.ask_treatments else [True],
						[True, False] if self.ask_outcomes else [True],
				        [True, False] if self.ask_polarities else [True]):

					corr = base_corr * (-1) ** (treated ^ result)

					answer = (-1) ** polarity

					yield {
						'given_info': util.pformat(self.given_info_template,
					                        treatment=self.treatment,
					                        outcome=self.outcome,
	                                        collider=collider,
	                                        baseline=baseline,
	                                        treated=treated,
	                                        result=result,
					                        polarity=polarity,
					                        correlation=corr,
					                        **labels),

						'question': util.pformat(self.question_template,
					                        treatment=self.treatment,
					                        outcome=self.outcome,
	                                        collider=collider,
	                                        baseline=baseline,
	                                        treated=treated,
	                                        result=result,
					                        polarity=polarity,
					                        **labels),

						'answer': util.pformat(self.answer_template,
					                        treatment=self.treatment,
					                        outcome=self.outcome,
	                                        collider=collider,
	                                        baseline=baseline,
	                                        treated=treated,
	                                        result=result,
					                        polarity=polarity,
						                       answer=answer,
						                       **labels),

						# 'background': background,

						'meta': {
							'treated': treated,
							'result': result,
							'baseline': baseline,
							'polarity': polarity,
							'collider': collider,
							'groundtruth': 'yes' if answer == 1 else 'no',
							'given_info': {util.pformat(self.symbolic_template,
					                        treatment=self.treatment,
					                        outcome=self.outcome,
	                                        collider=collider,
	                                        baseline=baseline,
	                                        treated=treated,
	                                        result=result,
					                        polarity=polarity,
					                        **labels): corr},
	                        'formal_form': util.pformat(self.formal_form,
					                        treatment=self.treatment,
					                        outcome=self.outcome,
	                                        collider=collider,
	                                        baseline=baseline,
	                                        treated=treated,
	                                        result=result,
					                        polarity=polarity,
					                        **labels),
							**meta,
						}
					}



	def meta_data(self, scm, labels):
		return {
			'query_type': self.name,
	        'rung': self.rung,

			'treatment': self.treatment,
			'outcome': self.outcome,
		}




































