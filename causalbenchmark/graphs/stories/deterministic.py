from typing import Dict, Sequence
from ... import util

from ..base import register_graph, hparam
from .phenomena import Phenomenon, Fork as NonDeterministicFork, Mediation as NonDeterministicMediation, \
	Diamond as NonDeterministicDiamond, DiamondCut as NonDeterministicDiamondCut, \
	Chain as NonDeterministicChain, InstrumentalVariable as NonDeterministicInstrumentalVariable, \
	Arrowhead as NonDeterministicArrowhead, Frontdoor as NonDeterministicFrontdoor, \
	Confounding as NonDeterministicConfounding, Collision as NonDeterministicCollision


class choice(hparam):
	pass



class DeterministicPhenomenon(Phenomenon):
	# background_edge_template = hparam('{parent} directly causes {children}.', inherit=True)

	equation_type = 'deterministic'

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._fix_params()


	@classmethod
	def _named_choices(cls):
		for key, val in cls.named_hyperparameters(hidden=True):
			if isinstance(val, choice):
				yield key, val


	@classmethod
	def spec_count(cls):
		return 2 ** len(list(cls._named_choices()))


	@classmethod
	def spawn_choices(cls):
		settings = [key for key, val in cls._named_choices()]
		for values in util.generate_all_bit_strings(len(settings), dtype=bool):
			yield dict(zip(settings, values.tolist()))



	def _fix_params(self):
		pass


	def brute_force(self):
		srcs = [node.name for node in self.source_variables()]
		if len(srcs) > 10:
			raise ValueError('Too many sources to brute force')
		bit_strings = util.generate_all_bit_strings(len(srcs)).astype(int)
		return dict(zip(srcs, bit_strings.T))


	description_template = hparam(None)
	# def verbalize_graph(self, labels) -> str:
	# 	if self.description_template is None:
	# 		raise NotImplementedError
	# 	return self.description_template.format(**labels)


	def verbalize_description(self, labels: Dict[str, str]) -> str:
		if self.description_template is None:
			raise NotImplementedError
		# return self.description_template.format(**labels)
		return util.pformat(self.description_template, **labels)


	def verbalize_background(self, labels) -> str:
		lines = []
		if len(self.background_prefix) > 0:
			lines.append(self.background_prefix)

		lines.append(self.verbalize_graph(labels))

		if self.unobserved_variables is not None:
			for v in self.unobserved_variables:
				lines.append(self.background_unobserved_template.format(vname=labels[f'{v}name']))

		return ' '.join(line[0].upper() + line[1:] for line in lines if len(line) > 0)


	@property
	def mechanism_formulas(self):
		return []


@register_graph('det-fork')
class Fork(NonDeterministicFork, DeterministicPhenomenon):
	conjunction = choice(False)


	def _fix_params(self):
		self.Y.param = [0, 0, 0, 1] if self.conjunction else [0, 1, 1, 1]


	@hparam
	def description_template(self):
		rel1 = 'and' if self.conjunction else 'or'
		return f'We know that {{X1_noun}} {rel1} {{V21_noun}} causes {{Y1_noun}}.'

	@property
	def mechanism_formulas(self):
		rel1 = 'and' if self.conjunction else 'or'
		return [f'{{Y}} = {{X}} {rel1} {{V2}}']



@register_graph('det-mediation')
class Mediation(NonDeterministicMediation, DeterministicPhenomenon):
	conjunction = choice(False)  # choose whether Y is a conjunction or disjunction of X and Z
	negation = choice(False) # choose whether Y is a conjunction or disjunction of X and Z


	def _fix_params(self):
		self.V2.param = [1, 0] if self.negation else [0, 1]
		self.Y.param = [0, 0, 0, 1] if self.conjunction else [0, 1, 1, 1]


	@hparam
	def description_template(self):
		rel1 = 'and' if self.conjunction else 'or'
		zval = '{V20_noun}' if self.negation else '{V21_noun}'
		return f'We know that {{X1_noun}} causes {zval}. {{X1_noun}} {rel1} {{V21_noun}} causes {{Y1_noun}}.'

	@property
	def mechanism_formulas(self):
		rel1 = 'and' if self.conjunction else 'or'
		zval = 'not ' if self.negation else ''
		return [f'{{V2}} = {zval}{{X}}', f'{{Y}} = {{X}} {rel1} {{V2}}']



@register_graph('det-collision')
class Collision(NonDeterministicCollision, DeterministicPhenomenon):
	conjunction = choice(False)  # choose whether Y is a conjunction or disjunction of X and Z


	def _fix_params(self):
		self.V3.param = [0, 0, 0, 1] if self.conjunction else [0, 1, 1, 1]


	@hparam
	def description_template(self):
		rel1 = 'and' if self.conjunction else 'or'
		return f'We know that {{X1_noun}} {rel1} {{Y1_noun}} causes {{V31_noun}}.'


	@property
	def mechanism_formulas(self):
		rel1 = 'and' if self.conjunction else 'or'
		return [f'{{V3}} = {{X}} {rel1} {{Y}}']




@register_graph('det-confounding')
class Confounding(NonDeterministicConfounding, DeterministicPhenomenon):
	conjunction = choice(False)  # choose whether Y is a conjunction or disjunction of X and Z
	negation = choice(False) # choose whether Y is a conjunction or disjunction of X and Z


	def _fix_params(self):
		self.X.param = [1, 0] if self.negation else [0, 1]
		self.Y.param = [0, 0, 0, 1] if self.conjunction else [0, 1, 1, 1]


	@hparam
	def description_template(self):
		rel1 = 'and' if self.conjunction else 'or'
		xval = '{X0_noun}' if self.negation else '{X1_noun}'
		return f'We know that {{V11_noun}} causes {xval}. {{V11_noun}} {rel1} {{X1_noun}} causes {{Y1_noun}}.'


	@property
	def mechanism_formulas(self):
		rel1 = 'and' if self.conjunction else 'or'
		zval = 'not ' if self.negation else ''
		return [f'{{X}} = {zval}{{V1}}', f'{{Y}} = {{V1}} {rel1} {{X}}']



@register_graph('det-diamond')
class Diamond(NonDeterministicDiamond, DeterministicPhenomenon):  # Create a class for the current SCM
	conjunction = choice(False)  # choose whether Y is a conjunction or disjunction of X and Z
	negation = choice(False)  # choose whether Y is a conjunction or disjunction of X and Z


	def _fix_params(self):
		self.V2.param = [0, 1]
		self.V3.param = [1, 0] if self.negation else [0, 1]
		self.Y.param = [0, 0, 0, 1] if self.conjunction else [0, 1, 1, 1]


	@hparam
	def description_template(self):
		rel1 = 'and' if self.conjunction else 'or'
		zval = '{V30_noun}' if self.negation else '{V31_noun}'
		return f'We know that {{X1_noun}} causes {{V21_noun}} and {zval}. {{V21_noun}} {rel1} {{V31_noun}} causes {{Y1_noun}}.'


	@property
	def mechanism_formulas(self):
		rel1 = 'and' if self.conjunction else 'or'
		zval = 'not ' if self.negation else ''
		return [f'{{V2}} = {{X}}', f'{{V3}} = {zval}{{V2}}', f'{{Y}} = {{V2}} {rel1} {{V3}}']


@register_graph('det-diamondcut')
class DiamondCut(NonDeterministicDiamondCut, DeterministicPhenomenon):  # Create a class for the current SCM
	conjunction = choice(False)  # choose whether Y is a conjunction or disjunction of X and Z
	negation = choice(False)  # choose whether Y is a conjunction or disjunction of X and Z


	def _fix_params(self):
		self.X.param = [0, 1]
		self.V3.param = [1, 0] if self.negation else [0, 1]
		self.Y.param = [0, 0, 0, 1] if self.conjunction else [0, 1, 1, 1]


	@hparam
	def description_template(self):
		rel1 = 'and' if self.conjunction else 'or'
		xval = '{X0_noun}' if self.negation else '{X1_noun}'
		return f'We know that {{V11_noun}} causes {xval} and {{V31_noun}}. {{X1_noun}} {rel1} {{V31_noun}} causes {{Y1_noun}}.'


	@property
	def mechanism_formulas(self):
		rel1 = 'and' if self.conjunction else 'or'
		zval = 'not ' if self.negation else ''
		return [f'{{X}} = {{V1}}', f'{{V3}} = {zval}{{X}}', f'{{Y}} = {{X}} {rel1} {{V3}}']



@register_graph('det-chain')
class Chain(NonDeterministicChain, DeterministicPhenomenon):
	negate_intermediate = choice(False)
	negate_outcome = choice(False)


	def _fix_params(self):
		self.V2.param = [1, 0] if self.negate_intermediate else [0, 1]
		self.Y.param = [1, 0] if self.negate_outcome else [0, 1]


	@hparam
	def description_template(self):
		v2val = '{V20_noun}' if self.negate_intermediate else '{V21_noun}'
		yval = '{Y0_noun}' if self.negate_outcome else '{Y1_noun}'
		return f'We know that {{X1_noun}} causes {v2val}, and we know that {{V21_noun}} causes {yval}.'


	@property
	def mechanism_formulas(self):
		v2val = 'not ' if self.negate_intermediate else ''
		yval = 'not ' if self.negate_outcome else ''
		return [f'{{V2}} = {v2val}{{X}}', f'{{Y}} = {yval}{{V2}}']


@register_graph('det-IV')
class InstrumentalVariable(NonDeterministicInstrumentalVariable, DeterministicPhenomenon):
	conjunction_treatment = choice(False)
	conjunction_outcome = choice(False)


	def _fix_params(self):
		self.X.param = [0, 0, 0, 1] if self.conjunction_treatment else [0, 1, 1, 1]
		self.Y.param = [0, 0, 0, 1] if self.conjunction_outcome else [0, 1, 1, 1]


	@hparam
	def description_template(self):
		rel1 = 'and' if self.conjunction_treatment else 'or'
		rel2 = 'and' if self.conjunction_outcome else 'or'
		return f'We know that {{V11_noun}} {rel1} {{V21_noun}} causes {{X1_noun}}. {{V11_noun}} {rel2} {{X1_noun}} causes {{Y1_noun}}.'


	@property
	def mechanism_formulas(self):
		rel1 = 'and' if self.conjunction_treatment else 'or'
		rel2 = 'and' if self.conjunction_outcome else 'or'
		return [f'{{X}} = {{V1}} {rel1} {{V2}}', f'{{Y}} = {{V1}} {rel2} {{X}}']


@register_graph('det-arrowhead')
class Arrowhead(NonDeterministicArrowhead, DeterministicPhenomenon):
	conjunction_intermediate = choice(False)
	conjunction_outcome = choice(False)


	def _fix_params(self):
		self.V3.param = [0, 0, 0, 1] if self.conjunction_intermediate else [0, 1, 1, 1]
		self.Y.param = [0, 0, 0, 0, 0, 0, 0, 1] if self.conjunction_outcome else [0, 1, 1, 1, 1, 1, 1, 1]


	@hparam
	def description_template(self):
		rel1 = 'and' if self.conjunction_intermediate else 'or'
		rel2 = 'and' if self.conjunction_outcome else 'or'
		return f'We know that {{X1_noun}} {rel1} {{V21_noun}} causes {{V31_noun}}. {{X1_noun}} {rel2} {{V21_noun}} {rel2} {{V31_noun}} causes {{Y1_noun}}.'


	@property
	def mechanism_formulas(self):
		rel1 = 'and' if self.conjunction_intermediate else 'or'
		rel2 = 'and' if self.conjunction_outcome else 'or'
		return [f'{{V3}} = {{X}} {rel1} {{V2}}', f'{{Y}} = {{X}} {rel2} {{V2}} {rel2} {{V3}}']



@register_graph('det-frontdoor')
class Frontdoor(NonDeterministicFrontdoor, DeterministicPhenomenon):
	negation = choice(False)
	conjunction = choice(False)


	def _fix_params(self):
		self.X.param = [0, 1]
		self.V3.param = [1, 0] if self.negation else [0, 1]
		self.Y.param = [0, 0, 0, 1] if self.conjunction else [0, 1, 1, 1]


	@hparam
	def description_template(self):
		rel1 = 'and' if self.conjunction else 'or'
		v3val = '{V30_noun}' if self.negation else '{V31_noun}'
		return f'We know that {{V11_noun}} causes {{X1_noun}}. {{X1_noun}} causes {v3val}. ' \
		       f'{{V11_noun}} {rel1} {{V31_noun}} causes {{Y1_noun}}.'


	@property
	def mechanism_formulas(self):
		rel1 = 'and' if self.conjunction else 'or'
		v3val = 'not ' if self.negation else ''
		return [f'{{X}} = {{V1}}', f'{{V3}} = {v3val}{{X}}', f'{{Y}} = {{V1}} {rel1} {{V3}}']



# class TwoCauses(Phenomenon,DeterministicSCM, name='det-twocauses'):  # Create a class for the current SCM
# 	conjunction = hparam(False)  # choose whether Y is a conjunction or disjunction of X and Z
#
# 	@hparam
# 	def description_template(self):
# 		rel1 = 'and' if self.conjunction else 'or'
# 		return f'We know that {{X1}} {rel1} {{Z1}} causes {{Y1}}.'
#
#
# 	@source('X')
# 	def X(self, N):
# 		return np.random.rand(N) < 0.5  # arbitrary Bernoulli(0.5) prior
#
#
# 	@source('Z')
# 	def Z(self, N):
# 		return np.random.rand(N) < 0.5  # arbitrary Bernoulli(0.5) prior
#
#
# 	@mechanism('Y')
# 	def Y(self, X, Z):
# 		return X & Z if self.conjunction else X | Z



# class Triangle(Phenomenon,DeterministicSCM, name='det-triangle'):
# 	conjunction = hparam(False)  # choose whether Y is a conjunction or disjunction of X and Z
# 	negation = hparam(False) # choose whether Y is a conjunction or disjunction of X and Z
#
#
# 	@hparam
# 	def description_template(self):
# 		rel1 = 'and' if self.conjunction else 'or'
# 		zval = '{Z0}' if self.negation else '{Z1}'
# 		return f'We know that {{X1}} causes {zval}. {{X1}} {rel1} {{Z1}} causes {{Yname}}.'
#
#
# 	@source('X')
# 	def X(self, N):
# 		return np.random.rand(N) < 0.5 # arbitrary Bernoulli(0.5) prior
#
#
# 	@mechanism('Z')
# 	def Z(self, X):
# 		return ~X if self.negation else X
#
#
# 	@mechanism('Y')
# 	def Y(self, X, Z):
# 		return X & Z if self.conjunction else X | Z



# class Diamond(Phenomenon,DeterministicSCM, name='det-diamond'): # Create a class for the current SCM
# 	conjunction = hparam(False) # choose whether Y is a conjunction or disjunction of X and Z
# 	negation = hparam(False) # choose whether Y is a conjunction or disjunction of X and Z
#
#
# 	@hparam
# 	def description_template(self):
# 		rel1 = 'and' if self.conjunction else 'or'
# 		zval = '{Z0}' if self.negation else '{Z1}'
# 		return f'We know that {{X1}} causes {zval} and {{W1}}. {{Z1}} {rel1} {{W1}} causes {{Yname}}.'
#
#
# 	@source('X')
# 	def X(self, N):
# 		return np.random.rand(N) < 0.5 # arbitrary Bernoulli(0.5) prior
#
#
# 	@mechanism('Z')
# 	def Z(self, X):
# 		return ~X if self.negation else X
#
#
# 	@mechanism('W')
# 	def W(self, X):
# 		return X
#
#
# 	@mechanism('Y')
# 	def Y(self, W, Z):
# 		return W & Z if self.conjunction else W | Z
#
#
# class DiamondCut(Phenomenon,DeterministicSCM, name='det-diamondcut'): # Create a class for the current SCM
# 	conjunction = hparam(False) # choose whether Y is a conjunction or disjunction of X and Z
# 	negation = hparam(False) # choose whether Y is a conjunction or disjunction of X and Z
#
#
# 	@hparam
# 	def description_template(self):
# 		rel1 = 'and' if self.conjunction else 'or'
# 		xval = '{X0}' if self.negation else '{X1}'
# 		return f'We know that {{Z1}} causes {xval} and {{W1}}. {{Z1}} {rel1} {{W1}} causes {{Yname}}.'
#
#
# 	@source('Z')
# 	def Z(self, N):
# 		return np.random.rand(N) < 0.5 # arbitrary Bernoulli(0.5) prior
#
#
# 	@mechanism('X')
# 	def X(self, Z):
# 		return ~Z if self.negation else Z
#
#
# 	@mechanism('W')
# 	def W(self, Z):
# 		return X
#
#
# 	@mechanism('Y')
# 	def Y(self, X, W):
# 		return X & Z if self.conjunction else W | X


