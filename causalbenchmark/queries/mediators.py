from typing import Generator
import numpy as np
from itertools import product
from .. import util
from .base import hparam, register_query
from .causal import ManualEstimandQueryType



class MediatorBasedQuery(ManualEstimandQueryType):
	name = None
	rung = 3
	formal_form = None

	question_template = None

	mediator_template = hparam('{{mediator}name}', inherit=True)

	ask_polarities = hparam(True, inherit=True)


	def verbalize_mediators(self, labels, mediators):
		return util.verbalize_list(util.pformat(self.mediator_template, mediator=m, **labels) for m in mediators)


	def meta_data(self, scm, labels, mediators=None):
		if mediators is None:
			mediators = list(scm.find_mediators(self.treatment, self.outcome))

		return {
			'query_type': self.name,
			'rung': self.rung,
	        'formal_form': util.pformat(self.formal_form, treatment=self.treatment, outcome=self.outcome,
	                                               mediators=mediators, **labels),

			'given_info': self.symbolic_given_info(scm),
			'estimand': self.symbolic_estimand(scm),

			'treatment': self.treatment,
			'outcome': self.outcome,
		}


	def _compute_groundtruth(self, scm, mediators=None):
		raise NotImplementedError


	def generate_questions(self, scm, labels, mediators=None):

		if mediators is None:
			mediators = list(scm.find_mediators(self.treatment, self.outcome))

		if not len(mediators):
			raise self._QueryFailedError('No mediators found')

		verbalized_mediators = self.verbalize_mediators(labels, mediators)

		try:
			meta = self.meta_data(scm, labels, mediators)
			given_info = self.verbalize_given_info(scm, labels)
			# background = self.verbalize_background(scm, labels)
		except AttributeError as e:
			raise self._QueryFailedError(str(e))

		lb, ub = self._compute_groundtruth(scm, mediators=mediators)

		sign = None if lb*ub <= 0 else (1 if ub > 0 else -1)

		for polarity in ([True, False] if self.ask_polarities else [True]):
			if sign is None or abs(lb) < 0.005:
				answer = -1
			else:
				answer = sign * (-1) ** (1-int(polarity))

			yield {
				'given_info': given_info,
				'question': util.pformat(self.question_template,
				                         treatment=self.treatment,
				                         outcome=self.outcome,
				                         mediators=verbalized_mediators,
				                         polarity=polarity,
				                         answer=answer,
				                         **labels),
				'answer': util.pformat(self.answer_template,
				                       treatment=self.treatment,
				                       outcome=self.outcome,
				                       mediators=verbalized_mediators,
				                       polarity=polarity,
				                       answer=answer,
				                       **labels),

				# 'background': background,

				'meta': {
					'mediators': mediators,
					'polarity': polarity,
					# 'groundtruth_bounds': [lb, ub],
					'groundtruth': lb,
					**meta,
				}
			}



@register_query('nde')
class NaturalDirectEffectQuery(MediatorBasedQuery):
	name = 'nde'
	rung = 3

	formal_form = 'E[{outcome}_{{{treatment}=1, {", ".join(m + "=0" for m in {mediators})}}} ' \
	              '- {outcome}_{{{treatment}=0, {", ".join(m + "=0" for m in {mediators})}}}]'

	question_template = hparam('If we disregard the mediation effect through {mediators}, '
	                           'would {{treatment}name} '
	                           '{"positively" if {polarity} else "negatively"} '
	                           'affect {{outcome}name}?')

	_known_estimands = {
		# chain
		"X->V2,V2->Y": None,
		# collision
		"X->V3,Y->V3": None,
		# frontdoor
		"V1->X,X->V3,V1->Y,V3->Y": None,
		# nondet-diamond
		"X->V3,X->V2,V2->Y,V3->Y": None,
		# IV
		"V1->X,V2->X,V1->Y,X->Y": ['[P(Y=1|V2=1)-P(Y=1|V2=0)]/[P(X=1|V2=1)-P(X=1|V2=0)]'],
		# arrowhead
		"X->V3,V2->V3,X->Y,V2->Y,V3->Y": [
			'\\sum_{V3=v} [\\sum_{V2=k}[P(Y=1|X=1,V3=v)-P(Y=1|X=0,V3=v)]*P(V3=v|X=0,V2=k)*P(V2=k)]'],
		# nondet-diamondcut
		"V1->V3,V1->X,X->Y,V3->Y": ['\\sum_{V1=v} P(V1=v)*[P(Y=1|V1=v,X=1) - P(Y=1|V1=v, X=0)]'],
		# confounding
		"V1->X,V1->Y,X->Y": ['\\sum_{V1=v} P(V1=v)*[P(Y=1|V1=v,X=1) - P(Y=1|V1=v, X=0)]'],
		# mediation
		"X->V2,X->Y,V2->Y": ['\\sum_{V2=v} P(V2=v|X=0)*[P(Y=1|X=1,V2=v) - P(Y=1|X=0, V2=v)]'],
	}

	_known_calculations = {
		"V1->X,V2->X,V1->Y,X->Y": '({given["p(Y | V2)"][1]:.2f} - {given["p(Y | V2)"][0]:.2f}) '
		                          '/ ({given["p(X | V2)"][1]:.2f} - {given["p(X | V2)"][0]:.2f}) = {result:.2f}',

		"X->V3,V2->V3,X->Y,V2->Y,V3->Y": '{given["p(V2)"][0]:.2f} * ({given["p(Y | X, V3)"][1][1]:.2f} - {given["p(Y | X, V3)"][1][0]:.2f}) '
		                                 '* {given["p(V3 | X, V2)"][0][1]:.2f} '
		                                 '+ {1-given["p(V2)"][0]:.2f} * ({given["p(Y | X, V3)"][0][1]:.2f} - {given["p(Y | X, V3)"][0][0]:.2f}) '
		                                 '* {given["p(V3 | X, V2)"][1][0]:.2f} '
		                                 '= {result:.2f}',

		"V1->V3,V1->X,X->Y,V3->Y": '{given["p(V1)"][0]:.2f} * ({given["p(Y | V1, X)"][1][1]:.2f} - {given["p(Y | V1, X)"][1][0]:.2f}) '
		                           '+ {1-given["p(V1)"][0]:.2f} * ({given["p(Y | V1, X)"][0][1]:.2f} - {given["p(Y | V1, X)"][0][0]:.2f}) '
		                           '= {result:.2f}',

		"V1->X,V1->Y,X->Y": '{given["p(V1)"][0]:.2f} * ({given["p(Y | V1, X)"][1][1]:.2f} - {given["p(Y | V1, X)"][1][0]:.2f}) '
		                    '+ {1-given["p(V1)"][0]:.2f} * ({given["p(Y | V1, X)"][0][1]:.2f} - {given["p(Y | V1, X)"][0][0]:.2f}) '
		                    '= {result:.2f}',

		"X->V2,X->Y,V2->Y": '{given["p(V2 | X)"][0]:.2f} * ({given["p(Y | X, V2)"][1][1]:.2f} - {given["p(Y | X, V2)"][1][0]:.2f}) '
		                    '+ {given["p(V2 | X)"][1]:.2f} * ({given["p(Y | X, V2)"][0][1]:.2f} - {given["p(Y | X, V2)"][0][0]:.2f}) '
		                    '= {result:.2f}',
	}

	_known_estimand_terms = {
		# chain
		"X->V2,V2->Y": ['P(Y|X)'],
		# collision
		"X->V3,Y->V3": None,
		# confounding
		"V1->X,V1->Y,X->Y": ['P(Y|V1,X)', 'P(V1)'],
		# mediation
		"X->V2,X->Y,V2->Y": ['P(Y|X,V2)', 'P(V2|X)'],
		# frontdoor
		"V1->X,X->V3,V1->Y,V3->Y": None,
		# IV
		"V1->X,V2->X,V1->Y,X->Y": ['P(Y|V2)', 'P(X|V2)'],
		# arrowhead
		"X->V3,V2->V3,X->Y,V2->Y,V3->Y": ['P(Y|X,V3)', 'P(V3|X,V2)', 'P(V2)'],
		# nondet-diamond
		"X->V3,X->V2,V2->Y,V3->Y": None,
		# nondet-diamondcut
		"V1->V3,V1->X,X->Y,V3->Y": ['P(Y|V1,X)', 'P(V1)'],
	}

	def _compute_groundtruth(self, scm, mediators=None):
		return scm.nde_bounds(self.outcome, self.treatment, mediators=mediators)



@register_query('nie')
class NaturalIndirectEffectQuery(MediatorBasedQuery):
	name = 'nie'
	rung = 3

	formal_form = 'E[{outcome}_{{{treatment}=0, {", ".join(m + "=1" for m in {mediators})}}} ' \
	              '- {outcome}_{{{treatment}=0, {", ".join(m + "=0" for m in {mediators})}}}]'


	question_template = hparam('Does {{treatment}name} '
	                           '{"positively" if {polarity} else "negatively"} '
	                           'affect {{outcome}name} through {mediators}?')

	_known_estimands = {
		# chain
		"X->V2,V2->Y": ['P(Y=1|X=1) - P(Y=1|X=0)'],
		# collision
		"X->V3,Y->V3": None,
		# confounding
		"V1->X,V1->Y,X->Y": None,
		# IV
		"V1->X,V2->X,V1->Y,X->Y": None,
		# nondet-diamondcut
		"V1->V3,V1->X,X->Y,V3->Y": None,
		# mediation
		"X->V2,X->Y,V2->Y": ['\\sum_{V2 = v} P(Y=1|X =0,V2 = v)*[P(V2 = v | X = 1) − P(V2 = v | X = 0)]'],
		# frontdoor
		"V1->X,X->V3,V1->Y,V3->Y": [
			'\\sum_{V3 = v} [P(V3 = v|X = 1) - P(V3 = v|X = 0)] * [\\sum_{X = h} P(Y = 1|X = h,V3 = v)*P(X = h)]'],
		# arrowhead
		"X->V3,V2->V3,X->Y,V2->Y,V3->Y": [
			'\\sum_{V3=v} [\\sum_{V2=k} P(Y=1|X=0,V3=v)*[P(V3=v|X=1,V2=k)-P(V3=v|X=0,V2=k)]*P(V2=k)]'],
		# nondet-diamond
		"X->V3,X->V2,V2->Y,V3->Y": ['P(Y=1|X=1) - P(Y=1|X=0)'],
	}

	_known_calculations = {

		"X->V2,V2->Y": '{given["p(Y | X)"][1]:.2f} - {given["p(Y | X)"][0]:.2f} = {result:.2f}',

		"X->V2,X->Y,V2->Y": '{given["p(V2 | X)"][1]:.2f} * ({given["p(Y | X, V2)"][0][1]:.2f} - {given["p(Y | X, V2)"][0][0]:.2f})'
		                    '+ {given["p(V2 | X)"][0]:.2f} * ({given["p(Y | X, V2)"][1][1]:.2f} - {given["p(Y | X, V2)"][1][0]:.2f})'
		                    '= {result:.2f}',

		"V1->X,X->V3,V1->Y,V3->Y": '{given["p(V3 | X)"][1]:.2f} - {given["p(V3 | X)"][0]:.2f} * '
		                           '({given["p(Y | X, V3)"][1][1]:.2f} * {given["p(X)"][0]:.2f} + '
		                           '{given["p(Y | X, V3)"][0][1]:.2f} * {1-given["p(X)"][0]:.2f})'
		                           '= {result:.2f}',

		"X->V3,V2->V3,X->Y,V2->Y,V3->Y": '{given["p(V2)"][0]:.2f} * {given["p(Y | X, V3)"][0][1]:.2f} '
		                                 '* ({given["p(V3 | X, V2)"][1][1]:.2f} - {given["p(V3 | X, V2)"][1][0]:.2f})'
		                                 '+ {1-given["p(V2)"][0]:.2f} * {given["p(Y | X, V3)"][0][1]:.2f} '
		                                 '* ({given["p(V3 | X, V2)"][0][1]:.2f} - {given["p(V3 | X, V2)"][0][0]:.2f})'
		                                 '= {result:.2f}',

		"X->V3,X->V2,V2->Y,V3->Y": '{given["p(Y | X)"][1]:.2f} - {given["p(Y | X)"][0]:.2f} = '
		                           '{given["p(Y | X)"][1] - given["p(Y | X)"][0]:.2f}',

	}

	_known_estimand_terms = {
		# chain
		"X->V2,V2->Y": ['P(Y|X)'],
		# collision
		"X->V3,Y->V3": None,
		# confounding
		"V1->X,V1->Y,X->Y": None,
		# mediation
		"X->V2,X->Y,V2->Y": ['P(Y|X,V2)', 'P(V2|X)'],
		# frontdoor
		"V1->X,X->V3,V1->Y,V3->Y": ['P(V3|X)', 'P(Y|X,V3)', 'P(X)'],
		# IV
		"V1->X,V2->X,V1->Y,X->Y": None,
		# arrowhead
		"X->V3,V2->V3,X->Y,V2->Y,V3->Y": ['P(Y|X,V3)', 'P(V3|X,V2)', 'P(V2)'],
		# nondet-diamond
		"X->V3,X->V2,V2->Y,V3->Y": ['P(Y|X)'],
		# nondet-diamondcut
		"V1->V3,V1->X,X->Y,V3->Y": None,
	}

	def _compute_groundtruth(self, scm, mediators=None):
		return scm.nie_bounds(self.outcome, self.treatment, mediators=mediators)











