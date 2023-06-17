from typing import Type, Any, Dict, Union, Iterator, Optional, Callable, Tuple, List, Sequence, Iterable, Set, Mapping

import numpy as np

from .. import util
from .base import AbstractQueryType, hparam, register_query



@register_query('det-counterfactual')
class DeterministicCounterfactualEffectQuery(AbstractQueryType):
	name = 'det-counterfactual'
	rung = 3
	formal_form = '{outcome}_{{{treatment}={action}}} = {polarity} | {evidence}'

	# question_template = 'Can we infer that {{outcome}{polarity}_sentence} ' \
	                    # 'had it been that {new_vals} instead of that {original_vals}?'

	question_template = 'Would {{outcome}{polarity}_sentence} if {new_vals} instead of {original_vals}?'
	_extra_info_template = 'We observed {evidence}.'
	answer_template = hparam('{{{0:"no", 1:"yes"}}[int({answer})]}', inherit=True)

	outcome = hparam('Y')
	treatment = hparam('X') # if set to None, all variables (except outcome) are considered as treatments

	def reasoning(self, scm, labels, entry):

		steps = {}

		steps['step0'] = 'Let ' + '; '.join(util.pformat('{v.name} = {{v.name}name}', v=v, **labels)
		                                  for v in scm.variables()) + '.'

		steps['step1'] = scm.symbolic_graph_structure()

		steps['step2'] = entry.get('meta', {}).get('formal_form', '')

		steps['step3'] = f'Solve for {self.outcome}, given the evidence and the action'

		given = entry.get('meta', {}).get('given_info', None)

		formulas = scm.mechanism_formulas

		terms = [f'{treatment} = {val}' for treatment, val in given.items()]

		var_ids = {v.name:v.name for v in scm.variables()}

		terms.extend(formula.format(**var_ids) for formula in formulas)

		steps['step4'] = '\n'.join(terms)

		gt = entry.get('meta', {}).get('groundtruth', None)

		end = [f for f in formulas if f.startswith('{Y} = ')]

		outcome = entry.get('meta', {}).get('outcome', None)
		if len(end):
			treatment = entry.get('meta', {}).get('treatment', None)
			action = entry.get('meta', {}).get('action', None)

			ctx = scm.context()
			ctx.update(given)
			ctx[treatment] = action
			ctx[outcome]

			steps['step5'] = 'Y = ' + end[0].format(**ctx)# util.pformat(calc, given=given, result=gt, **labels)

		else:
			steps['step5'] = f'{outcome} = {gt}'

		steps['end'] = str(int(gt)) # f'{gt:.2f} > 0' if gt > 0 else f'{gt:.2f} < 0'

		return steps



	def verbalize_evidence(self, labels, evidence: Dict[str, int]):
		if len(evidence) == 0:
			return ''
		lines = [labels[f'{name}{int(val)}_sentence'] for name, val in evidence.items()]
		evidence = util.verbalize_list(lines)
		return self._extra_info_template.format(evidence=evidence)


	def verbalize_action(self, labels, treatment, val):
		return labels[f'{treatment}{int(val)}_noun']


	def meta_data(self, scm, labels):
		return {
			'query_type': self.name,
	        'rung': self.rung,
	        # 'formal_form': self.formal_form.format(treatment=self.treatment, outcome=self.outcome),
			#
			# 'given_info': self.symbolic_given_info(scm),
			#
			# 'treatment': self.treatment,
			# 'outcome': self.outcome,
		}


	def get_formal_form(self, scm, labels):
		raise NotImplementedError


	def generate_questions(self, scm, labels):
		meta = self.meta_data(scm, labels)
		description = scm.verbalize_description(labels)

		if len(scm.get_variable(self.outcome).parents) == 0:
			raise self._QueryFailedError(f'{self.outcome} variable must have parents')

		if self.treatment is None:
			raise NotImplementedError # TODO: implement this
		treatments = [v.name for v in scm.variables() if v.name != self.outcome] \
			if self.treatment is None else [self.treatment]

		sources = [v.name for v in scm.variables() if v.name != self.outcome and not len(v.parents)]


		# given_info = scm.verbalize_description(labels)

		for treatment in treatments:
			evidence = [v for v in sources if v != treatment]

			names = [*evidence, treatment]
			samples = util.generate_all_bit_strings(len(names)).astype(int)

			ctx = scm.context()
			ctx.update(dict(zip(names, samples.T)))
			sols = ctx['Y']

			for sample, sol in zip(samples, sols.astype(bool)):
				evidence_values = dict(zip(evidence, sample.tolist()))

				verbalized_evidence = self.verbalize_evidence(labels, evidence_values)
				given_info = ' '.join([description,
				                       verbalized_evidence]) if len(verbalized_evidence) else description

				action = int(sample[-1])

				action_msg = self.verbalize_action(labels, treatment, sample[-1])
				original_msg = self.verbalize_action(labels, treatment, not sample[-1])

				for polarity, sol_int in zip([1, 0], [sol, not sol]):

					question = util.pformat(self.question_template,
					                        treatment=self.treatment,
					                        outcome=self.outcome,
					                        polarity=polarity,
					                        evidence=evidence_values,
					                        new_vals=action_msg,
					                        original_vals=original_msg,
					                        **labels)

					answer = util.pformat(self.answer_template,
					                        treatment=self.treatment,
					                        outcome=self.outcome,
					                        polarity=polarity,
					                        evidence=evidence_values,
					                        new_vals=action_msg,
					                        original_vals=original_msg,
					                      answer=sol_int,
					                      **labels)

					# question = self._template.format(Y_int=labels[f'Y{polarity}_noun'],
					#                                  cou_var_cou_val=action_msg,
					#                                  cou_val_original=original_msg,
					#                                  **labels)

					# answer = self._answer_template.format(sol='yes' if sol_int else 'no', **labels)

					# yield {'question': question,
					#        'answer': answer,
					#        'background': background,
					#        'given_info': given_info,
					#        'meta': {'evidence': evidence_values,
					#                 'treatment': treatment,
					#                 'action': action,
					#                 'polarity': polarity,
					#                 'answer': int(sol_int),
					#                 }
					#        }


					yield {
						'given_info': given_info,
						'question': question,
						'answer': answer,

						'meta': {
							'given_info': evidence_values,
							'formal_form': util.pformat(self.formal_form,
							                            treatment=treatment,
							                            outcome=self.outcome,
							                            evidence=', '.join([f'{k}={v}'
							                                                for k, v in evidence_values.items()]),
							                            action=action,
							                            polarity=polarity,
							                            **labels),
							# 'formal_form': self.get_formal_form(scm, labels),
			                'treatment': treatment,
							'outcome': self.outcome,
			                'action': action,
			                'polarity': polarity,
			                'groundtruth': int(sol_int),
							**meta,
						}
					}




class DeterministicInterventionEffectQuery(AbstractQueryType):
	_template = 'If we force {intervention}, will {Y_int} {universal} happend?'
	_answer_template = '{sol}'


	def __init__(self, templates=None, **kwargs):
		if templates is None:
			templates = {'question': self._template, 'answer': self._answer_template}
		super().__init__(**kwargs)
		self._templates = templates


	outcome = hparam('Y')
	treatment = hparam('X') # if set to None, all variables are considered as treatments


	def generate_questions(self, scm, labels):
		background = scm.verbalize_graph(labels)

		assert len(scm.get_variable(self.outcome).parents) > 0, 'outcome variable must have parents'

		treatments = [v.name for v in scm.variables() if v.name != self.outcome] \
			if self.treatment is None else [self.treatment]

		sources = [v.name for v in scm.variables() if v.name != self.outcome and not len(v.parents)]

		for treatment in treatments:
			evidence = [v for v in sources if v != treatment]

			names = [treatment, *evidence]
			samples = util.generate_all_bit_strings(len(names)).astype(int)

			ctx = scm.context()
			ctx.update(dict(zip(names, samples.T)))
			all_sols = ctx['Y']

			for intv in [False, True]:
				intervention = labels[f'{treatment}{int(intv)}_noun']

				sols = all_sols[samples[:, 0] == intv]

				for polarity, sol in zip([1, 0], [sols, ~sols]):
					Y_int = labels[f'Y{polarity}_noun']
					for universal, sol_int in zip(['always', 'never'], [np.all(sol), ~np.any(sol)]):
						question = self._template.format(Y_int=Y_int,
						                                universal=universal,
						                                intervention=intervention,
						                                **labels)
						answer = self._answer_template.format(sol='yes' if sol_int else 'no', **labels)
						yield {'question': question,
						       'answer': answer,
						       'background': background,
						       'given_info': '', # no evidence needed
						       'meta': {'treatment': treatment,
						                'intervention': int(intv),
						                'polarity': polarity,
						                'evidence': evidence,
						                'universal': universal,
						                'answer': int(sol_int),
						                }}


