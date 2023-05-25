from typing import Generator
import numpy as np
from itertools import product
from .. import util
from .base import AbstractQueryType, hparam, register_query



@register_query('marginal')
class MarginalQuery(AbstractQueryType):
	name = 'marginal'
	rung = 1
	formal_form = 'P({outcome})'

	# question_template = hparam('Is the overall likelihood of {{subject}{polarity}_noun} greater than chance?')
	question_template = hparam('Is {{outcome}{int(treated)}_noun} {"more" if {polarity} else "less"} likely '
	                           'than {{outcome}{int(not treated)}_noun} overall?')

	answer_template = hparam('{{{-1:"no", 0:"unknown", 1:"yes"}}[int({answer})]}')

	prior_template = hparam('The overall probability of {{treatment}1_noun} is {prior:.0%}.')
	conditional_template = hparam('For {{treatment}0_wheresentence}, the probability of {{outcome}1_noun} is {cond[0]:.0%}. '
	                              'For {{treatment}1_wheresentence}, the probability of {{outcome}1_noun} is {cond[1]:.0%}.')

	treatment = hparam('X')
	outcome = hparam('Y')

	ask_polarities = hparam(True)
	ask_treatments = hparam(False)


	def reasoning(self, scm, labels, entry):

		steps = {}

		steps['step0'] = 'Let ' + '; '.join(util.pformat('{v.name} = {{v.name}name}', v=v, **labels)
		                                  for v in scm.variables()) + '.'

		steps['step1'] = scm.symbolic_graph_structure()


		steps['step2'] = entry.get('meta', {}).get('formal_form', '')

		steps['step3'] = f'P({self.outcome} | {self.treatment}=1)*P({self.treatment}=1) ' \
		                 f'+ P({self.outcome} | {self.treatment}=0)*P({self.treatment}=0)'

		given = entry.get('meta', {}).get('given_info', None)

		terms = util.parse_given_info(given)
		steps['step4'] = '\n'.join(terms)


		calc = '{given["P(X)"]:.2f}*{given["P(Y | X)"][1]:.2f} - {1-given["P(X)"]:.2f}*{given["P(Y | X)"][0]:.2f} ' \
		       '= {result:.2f}'

		gt = entry.get('meta', {}).get('groundtruth', None)

		steps['step5'] = util.pformat(calc, given=given, result=gt, **labels)

		if abs(gt) < 0.005:
			steps['end'] = '0.00 = 0'
		else:
			steps['end'] = f'{gt:.2f} > 0' if gt > 0 else f'{gt:.2f} < 0'

		return steps



	def verbalize_given_info(self, scm, labels, cond, prior, **details):
		return ' '.join([util.pformat(self.prior_template, treatment=self.treatment, outcome=self.outcome,
		                              prior=prior, **labels),
		                 util.pformat(self.conditional_template, treatment=self.treatment, outcome=self.outcome,
		                              cond=cond, **labels)])


	def symbolic_given_info(self, scm, labels, cond, prior, **details):
		return {
			f'P({self.treatment})': prior,
			f'P({self.outcome} | {self.treatment})': cond,
		}


	def generate_questions(self, scm, labels):

		# answer = prior*cond1 + (1-prior)*cond0
		
		meta = self.meta_data(scm, labels)
		# background = self.verbalize_background(scm, labels)
		
		prior_lb, prior_ub = scm.marginal_bounds(self.treatment)
		if not np.isclose(prior_lb, prior_ub):
			raise NotImplementedError
		prior = prior_lb

		cond1_lb, cond1_ub = scm.marginal_bounds(self.outcome, **{self.treatment: 1})
		if not np.isclose(cond1_lb, cond1_ub):
			raise NotImplementedError
		cond1 = cond1_lb

		cond0_lb, cond0_ub = scm.marginal_bounds(self.outcome, **{self.treatment: 0})
		if not np.isclose(cond0_lb, cond0_ub):
			raise NotImplementedError
		cond0 = cond0_lb
		
		cond = [cond0, cond1]
		given_info = self.verbalize_given_info(scm, labels, cond=cond, prior=prior)
		symbolic_given_info = self.symbolic_given_info(scm, labels, cond=cond, prior=prior)

		lb, ub = scm.marginal_bounds(self.outcome)
		slb = 2*lb - 1
		sub = 2*ub - 1

		sign = None if slb*sub <= 0 else (1 if sub > 0 else -1)

		for treated, polarity in product([True, False] if self.ask_treatments else [True],
		                                [True, False] if self.ask_polarities else [True]):
			if sign is None or abs(lb) < 0.005:
				answer = -1 # default answer is no
			else:
				answer = sign * (-1) ** (1-int(polarity) + 1-int(treated))

			yield {
				'given_info': given_info,
				'question': util.pformat(self.question_template,
				                         treatment=self.treatment,
				                         outcome=self.outcome,
				                         treated=treated,
				                         polarity=polarity,
				                         answer=answer,
				                         **labels),
				'answer': util.pformat(self.answer_template,
				                         treatment=self.treatment,
				                         outcome=self.outcome,
				                       treated=treated,
				                       polarity=polarity,
				                       answer=answer,
				                       **labels),

				# 'background': background,

				'meta': {
					'given_info': symbolic_given_info,
					'treated': treated,
					'polarity': polarity,
					# 'groundtruth_bounds': [lb, ub],
					'groundtruth': lb,
					**meta,
				}
			}


	def meta_data(self, scm, labels): # returns meta['query']
		return {
			'query_type': self.name,
	        'rung': self.rung,
	        'formal_form': self.formal_form.format(treatment=self.treatment, outcome=self.outcome),

	        'treatment': self.treatment,
			'outcome': self.outcome,
		}



@register_query('correlation')
class CorrelationQuery(AbstractQueryType):
	name = 'correlation'
	rung = 1
	formal_form = 'P({outcome} | {treatment})'

	question_template = hparam('Is the chance of {{outcome}{int(result)}_noun} '
	                           '{"larger" if {polarity} else "smaller"} when observing '
	                           '{{treatment}{int(treated)}_noun}?')

	answer_template = hparam('{{{-1:"no", 0:"unknown", 1:"yes"}}[int({answer})]}')

	marginal_template = hparam('The overall probability of {{treatment}1_noun} is {marginal:.0%}.')
	joint_template = hparam('The probability of {{treatment}0_noun} and {{outcome}1_noun} is {joint[0]:.0%}. '
	                        'The probability of {{treatment}1_noun} and {{outcome}1_noun} is {joint[1]:.0%}.')

	treatment = hparam('X')
	outcome = hparam('Y')

	ask_polarities = hparam(True)
	ask_treatments = hparam(False)
	ask_results = hparam(False)


	def reasoning(self, scm, labels, entry):

		steps = {}

		steps['step0'] = 'Let ' + '; '.join(util.pformat('{v.name} = {{v.name}name}', v=v, **labels)
		                                  for v in scm.variables()) + '.'

		steps['step1'] = scm.symbolic_graph_structure()


		steps['step2'] = entry.get('meta', {}).get('formal_form', '')

		steps['step3'] = 'P(X = 1, Y = 1)/P(X = 1) - P(X = 0, Y = 1)/P(X = 0)'

		given = entry.get('meta', {}).get('given_info', None)

		terms = util.parse_given_info(given)
		steps['step4'] = '\n'.join(terms)


		calc = '{given["P(Y=1, X=1)"]:.2f}/{given["P(X=1)"]:.2f} ' \
		       '- {given["P(Y=1, X=0)"]:.2f}/{1-given["P(X=1)"]:.2f} ' \
		       '= {result:.2f}'

		gt = entry.get('meta', {}).get('groundtruth', None)

		steps['step5'] = util.pformat(calc, given=given, result=gt, **labels)

		if abs(gt) < 0.005:
			steps['end'] = '0.00 = 0'
		else:
			steps['end'] = f'{gt:.2f} > 0' if gt > 0 else f'{gt:.2f} < 0'

		return steps


	def verbalize_given_info(self, scm, labels, joint, marginal, **details):
		return ' '.join([util.pformat(self.marginal_template, treatment=self.treatment, outcome=self.outcome,
		                              marginal=marginal, **labels),
		                 util.pformat(self.joint_template, treatment=self.treatment, outcome=self.outcome,
		                              joint=joint, **labels)])

	def symbolic_given_info(self, scm, labels, joint, marginal, **details):
		return {
			f'P({self.treatment}=1)': marginal,
			f'P({self.outcome}=1, {self.treatment}=0)': joint[0],
			f'P({self.outcome}=1, {self.treatment}=1)': joint[1],
		}

	def generate_questions(self, scm, labels):

		# answer = prior*cond1 + (1-prior)*cond0

		meta = self.meta_data(scm, labels)
		# background = self.verbalize_background(scm, labels)

		marginal_lb, marginal_ub = scm.marginal_bounds(self.treatment)
		if not np.isclose(marginal_lb, marginal_ub):
			raise NotImplementedError
		marginal = marginal_lb

		joint0 = scm.probability(**{self.treatment: 0, self.outcome: 1})
		joint1 = scm.probability(**{self.treatment: 1, self.outcome: 1})
		joint = [joint0, joint1]

		given_info = self.verbalize_given_info(scm, labels, joint=joint, marginal=marginal)
		symbolic_given_info = self.symbolic_given_info(scm, labels, joint=joint, marginal=marginal)

		x1 = joint1 / marginal
		x0 = joint0 / (1 - marginal)
		corr = x1 - x0

		sign = None if corr == 0 else (1 if corr > 0 else -1)

		for treated, result, polarity in product([True, False] if self.ask_treatments else [True],
				                                 [True, False] if self.ask_results else [True],
				                                 [True, False] if self.ask_polarities else [True]):
			if sign is None or abs(corr) < 0.005:
				answer = -1  # default answer is no
			else:
				answer = sign * (-1) ** (1 - int(polarity) + 1 - int(treated) + 1 - int(result))

			yield {
				'given_info': given_info,
				'question': util.pformat(self.question_template,
				                         treatment=self.treatment,
				                         outcome=self.outcome,
				                         treated=treated,
				                         polarity=polarity,
				                         result=result,
				                         answer=answer,
				                         **labels),
				'answer': util.pformat(self.answer_template,
				                       treatment=self.treatment,
				                       outcome=self.outcome,
				                       treated=treated,
				                       polarity=polarity,
				                       result=result,
				                       answer=answer,
				                       **labels),

				# 'background': background,

				'meta': {
					'given_info': symbolic_given_info,
					'treated': treated,
					'polarity': polarity,
					# 'groundtruth_bounds': [corr, corr],
					'groundtruth': corr,
					**meta,
				}
			}

	def meta_data(self, scm, labels):  # returns meta['query']
		return {
			'query_type': self.name,
			'rung': self.rung,
			'formal_form': self.formal_form.format(treatment=self.treatment, outcome=self.outcome),

			'treatment': self.treatment,
			'outcome': self.outcome,
		}




@register_query('exp_away')
class ExplainAwayQuery(AbstractQueryType):
	name = 'exp_away'
	rung = 1
	formal_form = 'P({outcome} = {int(result)} | {treatment} = {int(treated)}, {collider} = {int(baseline)}] ' \
	              '- P({outcome} = {int(result)} | {collider} = {int(baseline)})'


	question_template = hparam('If we look at {{collider}{int(baseline)}_wheresentence}, '
	                           'does the chance of {{outcome}{int(result)}_noun} '
	                           '{"increase" if {polarity} else "decrease"} '
	                           'when {{treatment}{int(treated)}_noun}?')

	answer_template = hparam('{{{-1:"no", 0:"unknown", 1:"yes"}}[int({answer})]}')

	treatment = hparam('X')
	outcome = hparam('Y')

	ask_all_colliders = hparam(False)

	ask_polarities = hparam(True)
	ask_treatments = hparam(False)
	ask_outcomes = hparam(False)
	ask_baselines = hparam(False)

	mechanism_template = hparam('For {{treatment}0_wheresentence} and {{collider}0_wherepartial}, '
	                             'the probability of {{outcome}1_noun} is {param[0,0]:.0%}. '
	                             'For {{treatment}0_wheresentence} and {{collider}1_wherepartial}, '
	                             'the probability of {{outcome}1_noun} is {param[0,1]:.0%}. '
	                             'For {{treatment}1_wheresentence} and {{collider}0_wherepartial}, '
	                             'the probability of {{outcome}1_noun} is {param[1,0]:.0%}. '
	                             'For {{treatment}1_wheresentence} and {{collider}1_wherepartial}, '
	                             'the probability of {{outcome}1_noun} is {param[1,1]:.0%}.',
	                             inherit=True)


	def verbalize_given_info(self, scm, labels, collider, param, prior, **details):

		treatment_prior = scm.verbalize_mechanism(self.treatment, labels)

		mech = util.pformat(self.mechanism_template,
		                    treatment=self.treatment,
		                    outcome=self.outcome,
		                    collider=collider,
		                    param=np.asarray(param),
		                    **labels)

		return f'{treatment_prior} {mech}'


	def symbolic_given_info(self, scm, labels, collider, param, prior, **details):
		return {
			'P({treatment})'.format(treatment=self.treatment): prior,
			'P({outcome} | {treatment}, {collider})'.format(outcome=self.outcome,
			                                                treatment=self.treatment,
			                                                collider=collider): param,

		}


	def reasoning(self, scm, labels, entry):

		steps = {}

		steps['step0'] = 'Let ' + '; '.join(util.pformat('{v.name} = {{v.name}name}', v=v, **labels)
		                                  for v in scm.variables()) + '.'

		steps['step1'] = scm.symbolic_graph_structure()


		steps['step2'] = entry.get('meta', {}).get('formal_form', '')

		steps['step3'] = 'P(Y=1 | X=1, V3=1) - (P(X=1) * P(Y=1 | X=1, V3=1) + P(X=0) * P(Y=1 | X=0, V3=1))'

		given = entry.get('meta', {}).get('given_info', None)

		terms = util.parse_given_info(given)
		steps['step4'] = '\n'.join(terms)

		calc = '{given["P(Y | X, V3)"][1][1]:.2f} ' \
		       '- ({given["P(X)"]:.2f}*{given["P(Y | X, V3)"][1][1]:.2f} ' \
		       '+ {1-given["P(X)"]:.2f}*{given["P(Y | X, V3)"][0][1]:.2f}) ' \
		       '= {result:.2f}'

		gt = entry.get('meta', {}).get('groundtruth', None)

		steps['step5'] = util.pformat(calc, given=given, result=gt, **labels)

		if abs(gt) < 0.005:
			steps['end'] = '0.00 = 0'
		else:
			steps['end'] = f'{gt:.2f} > 0' if gt > 0 else f'{gt:.2f} < 0'

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

		prior = scm.marginals()[self.treatment]

		for collider in colliders:
			param = [[scm.marginals(**{self.treatment: i, collider: j})[self.outcome]
			          for j in [0, 1]] for i in [0, 1]]

			given_info = self.verbalize_given_info(scm, labels, collider, param, prior)
			symbolic_given_info = self.symbolic_given_info(scm, labels, collider, param, prior)

			for baseline, treated in product([True, False] if self.ask_baselines else [True],
			                                 [True, False] if self.ask_treatments else [True]):

				now = scm.marginals(**{self.treatment: treated, collider: baseline})[self.outcome]
				prev = scm.marginals(**{collider: baseline})[self.outcome]
				exp_away_effect = now - prev

				base = 0 if exp_away_effect == 0 else 1 if exp_away_effect > 0 else -1

				for result, polarity in product(
						[True, False] if self.ask_outcomes else [True],
				        [True, False] if self.ask_polarities else [True]):

					answer = base * (-1) ** (polarity ^ result)

					yield {
						'given_info': given_info,

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
							'given_info': symbolic_given_info,
							'collider': collider,
							'groundtruth': exp_away_effect,
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








































