from typing import Generator
import numpy as np
from itertools import product
from .. import util
from .base import MetricQueryType, hparam, register_query
from .causal import ManualEstimandQueryType



@register_query('ett')
class EffectTreatmentTreatedQuery(ManualEstimandQueryType):
	name = 'ett'
	rung = 3
	formal_form = 'E[Y_{{X = 1}} - Y_{{X = 0}} | X = 1]'

	question_template = hparam('For {{treatment}{int(treated)}_wheresentence}, would it be '
	                           '{"more" if {polarity} else "less"} likely to see {{outcome}{int(result)}_noun} '
	                           '{{treatment}{int(not treated)}_sentence_condition.split(" instead of")[0]}?')
								# last part is equivalent to the "_sentence_condition_chop" option

	ask_treatments = hparam(True)
	ask_polarities = hparam(True)
	ask_outcomes = hparam(False)

	_known_estimands = {
		# chain
		"X->V2,V2->Y": ['P(Y=1|X=1) - P(Y=1|X=0)'],
		"X->Y,V2->Y": ['P(Y=1|X=1) - P(Y=1|X=0)'],
		# collision
		"X->V3,Y->V3": None,
		# mediation
		"X->V2,X->Y,V2->Y": ['P(Y=1|X=1) - P(Y=1|X=0)'],
		# IV
		"V1->X,V2->X,V1->Y,X->Y": None,
		# arrowhead
		"X->V3,V2->V3,X->Y,V2->Y,V3->Y": ['P(Y=1|X=1) - P(Y=1|X=0)'],
		# nondet-diamond
		"X->V3,X->V2,V2->Y,V3->Y": ['P(Y=1|X=1) - P(Y=1|X=0)'],
		# nondet-diamondcut
		"V1->V3,V1->X,X->Y,V3->Y": ['\\sum_{V3=v} P(V3=v|X=1)*[P(Y=1|X=1,V3=v) - P(Y=1|X=0,V3=v)]'],
		# confounding
		"V1->X,V1->Y,X->Y": ['\\sum_{V1=v} P(V1=v|X=1)*[P(Y=1|V1=v,X=1) - P(Y=1|V1=v, X=0)]'],
		# frontdoor
		"V1->X,X->V3,V1->Y,V3->Y": ['\\sum_{V3=v}P(Y=1|X=1,V3=v)*[P(V3=v|X=1)-P(V3=v|X=0)]'],
			# '\\sum_{V3 = v} [P(V3 = v|X = 1) - P(V3 = v|X = 0)] * [\\sum_{X = h} P(Y = 1|X = h,V3 = v)*P(X = h)]'],
	}


	_known_calculations = {
		"X->V2,V2->Y": '{given["p(Y | X)"][1]:.2f} - {given["p(Y | X)"][0]:.2f} = {result:.2f}',
		"X->Y,V2->Y": '{given["p(Y | X)"][1]:.2f} - {given["p(Y | X)"][0]:.2f} = {result:.2f}',
		"X->V2,X->Y,V2->Y": '{given["p(Y | X)"][1]:.2f} - {given["p(Y | X)"][0]:.2f} = {result:.2f}',
		"X->V3,V2->V3,X->Y,V2->Y,V3->Y": '{given["p(Y | X)"][1]:.2f} - {given["p(Y | X)"][0]:.2f} = {result:.2f}',
		"X->V3,X->V2,V2->Y,V3->Y": '{given["p(Y | X)"][1]:.2f} - {given["p(Y | X)"][0]:.2f} = {result:.2f}',

		"V1->X,V1->Y,X->Y": '{1-given["p(V1)"][0]:.2f} '
		                    '* ({given["p(Y | V1, X)"][0][1]:.2f} - {given["p(Y | V1, X)"][0][0]:.2f}) '
		                    '+ {given["p(V1)"][0]:.2f} '
		                    '* ({given["p(Y | V1, X)"][1][1]:.2f} - {given["p(Y | V1, X)"][1][0]:.2f}) '
		                    '= {result:.2f}',

		"V1->X,X->V3,V1->Y,V3->Y": '{given["p(Y | X, V3)"][1][1]:.2f} * ({given["p(V3 | X)"][1]:.2f} - {given["p(V3 | X)"][0]:.2f}) '
		                        '+ {given["p(Y | X, V3)"][0][1]:.2f} * ({1-given["p(V3 | X)"][1]:.2f} - {1-given["p(V3 | X)"][0]:.2f}) '
		                        '= {result:.2f}',

		"V1->V3,V1->X,X->Y,V3->Y": '{given["p(V3 | X)"][1]:.2f} * ({given["p(Y | X, V3)"][1][1]:.2f} - {given["p(Y | X, V3)"][1][0]:.2f}) '
		                           '+ {given["p(V3 | X)"][0]:.2f} * ({given["p(Y | X, V3)"][0][1]:.2f} - {given["p(Y | X, V3)"][0][0]:.2f}) '
		                           '= {result:.2f}',

	}

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
		"V1->X,X->V3,V1->Y,V3->Y": ['P(Y|X,V3)', 'P(V3|X)'],
		# IV
		"V1->X,V2->X,V1->Y,X->Y": None,
		# arrowhead
		"X->V3,V2->V3,X->Y,V2->Y,V3->Y": ['P(Y|X)'],
		# nondet-diamond
		"X->V3,X->V2,V2->Y,V3->Y": ['P(Y|X)'],
		# nondet-diamondcut
		"V1->V3,V1->X,X->Y,V3->Y": ['P(Y|X,V3)', 'P(V3|X)'],
	}

	def generate_questions(self, scm, labels):
		try:
			meta = self.meta_data(scm, labels)
			given_info = self.verbalize_given_info(scm, labels)
			# background = self.verbalize_background(scm, labels)
		except AttributeError as e:
			raise self._QueryFailedError(str(e))

		for treated in ([True, False] if self.ask_treatments else [True]):

			lb, ub = scm.ett_bounds(self.outcome, self.treatment, treated=treated)

			sign = None if lb*ub <= 0 else (1 if ub > 0 else -1)

			for result, polarity in product([True, False] if self.ask_outcomes else [True],
			                                [True, False] if self.ask_polarities else [True]):
				if sign is None:
					answer = 0
				elif abs(lb) < 0.005:
					answer = -1
				else:
					answer = sign * (-1) ** (1 + 1-int(result) + 1-int(polarity))

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

					# 'background': background,

					'meta': {
						'treated': treated,
						'result': result,
						'polarity': polarity,
					# 'groundtruth_bounds': [lb, ub],
					'groundtruth': lb,
						**meta,
					}
				}





















































