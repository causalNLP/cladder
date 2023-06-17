from typing import Type, Any, Dict, Union, Iterator, Optional, Callable, Tuple, FrozenSet, List, Sequence, Iterable, Set, Mapping
import numpy as np

from functools import lru_cache

from ... import util

from ..base import register_graph, AbstractVariable
from ..bayes import BayesNet, node, hparam


class MissingEstimandError(NotImplementedError):
	pass


class Phenomenon(BayesNet):
	# _all_backdoor_adjustment_sets = None

	# def __init__(self, *args, **kwargs):
	# 	super().__init__(*args, **kwargs)
	# 	self.config = {} if self._graph_name is None else self._load_graph_config(self._graph_name)

	# def backdoor_adjustment_sets(self, outcome: str, treatment: str) -> Iterator[FrozenSet[str]]:
	# 	"""Returns all backdoor adjustment sets for the given outcome and treatment."""
	# 	if not (outcome == 'Y' and treatment == 'X'):
	# 		raise NotImplementedError(f'Adjustment sets are only implemented for Y and X '
	# 		                          f'instead of {outcome} and {treatment}')
	# 	# for adj in self._all_backdoor_adjustment_sets:
	# 	# 	yield frozenset(adj)
	# 	yield from self._all_backdoor_adjustment_sets


	# @staticmethod
	# def _load_graph_config(graph_name):
	# 	return util.load_graph_config(graph_name)


	# def symbolic_ate_estimand(self, outcome: str, treatment: str, *, treated: bool = True) -> str:
	# 	if outcome != 'Y' or treatment != 'X':
	# 		raise NotImplementedError(f'Estimands are only implemented for Y and X '
	# 		                          f'instead of {outcome} and {treatment}')
	#
	# 	est = self.config.get('ate_estimand', None)
	# 	if est is None:
	# 		return super().symbolic_ate_estimand(outcome, treatment, treated=treated)
	# 	return est


	# def symbolic_ett_estimand(self, outcome: str, treatment: str, *, treated: bool = True) -> str:
	# 	if outcome != 'Y' or treatment != 'X':
	# 		raise NotImplementedError(f'Estimands are only implemented for Y and X '
	# 		                          f'instead of {outcome} and {treatment}')
	#
	# 	est = self.config.get('ett_estimand', None)
	# 	if est is None:
	# 		return super().symbolic_ett_estimand(outcome, treatment, treated=treated)
	# 	return est


	# def symbolic_nde_estimand(self, outcome: str, treatment: str, *, mediators: Optional[Sequence[str]] = None,
	#                            treated: bool = True) -> str:
	# 	if outcome != 'Y' or treatment != 'X':
	# 		raise NotImplementedError(f'Estimands are only implemented for Y and X '
	# 		                          f'instead of {outcome} and {treatment}')
	#
	# 	est = self.config.get('nde_estimand', None)
	# 	if est is None:
	# 		return super().symbolic_nde_estimand(outcome, treatment, mediators=mediators, treated=treated)
	# 	return est


	# def symbolic_nie_estimand(self, outcome: str, treatment: str, *, mediators: Optional[Sequence[str]] = None,
	#                            treated: bool = True) -> str:
	# 	if outcome != 'Y' or treatment != 'X':
	# 		raise NotImplementedError(f'Estimands are only implemented for Y and X '
	# 		                          f'instead of {outcome} and {treatment}')
	#
	# 	est = self.config.get('nie_estimand', None)
	# 	if est is None:
	# 		return super().symbolic_nie_estimand(outcome, treatment, mediators=mediators, treated=treated)
	# 	return est


	# _MissingEstimandError = MissingEstimandError

	# def symbolic_ate_variables(self, outcome: str, treatment: str) -> List[str]:
	# 	terms = self.config.get('ate_nec', None)
	# 	if terms is None or outcome != 'Y' or treatment != 'X':
	# 		raise self._MissingEstimandError(f'ate unknown for treatment {treatment} and outcome {outcome}')
	#
	# 	given = [self._mechanism_str_to_variable(term) for term in terms.split(';')]
	# 	return given


	# def symbolic_ett_variables(self, outcome: str, treatment: str) -> List[str]:
	# 	terms = self.config.get('ett_nec', None)
	# 	if terms is None or outcome != 'Y' or treatment != 'X':
	# 		raise self._MissingEstimandError(f'ate unknown for treatment {treatment} and outcome {outcome}')
	#
	# 	given = [self._mechanism_str_to_variable(term) for term in terms.split(';')]
	# 	return given


	# def symbolic_nde_variables(self, outcome: str, treatment: str, *,
	#                            mediators: Optional[Sequence[str]] = None) -> List[str]:
	# 	terms = self.config.get('nde_nec', None)
	# 	if terms is None or outcome != 'Y' or treatment != 'X':
	# 		raise self._MissingEstimandError(f'nde unknown for treatment {treatment} and outcome {outcome}')
	#
	# 	given = [self._mechanism_str_to_variable(term) for term in terms.split(';')]
	# 	return given


	# def symbolic_nie_variables(self, outcome: str, treatment: str, *,
	#                            mediators: Optional[Sequence[str]] = None) -> List[str]:
	# 	terms = self.config.get('nie_nec', None)
	# 	if terms is None or outcome != 'Y' or treatment != 'X':
	# 		raise self._MissingEstimandError(f'nie unknown for treatment {treatment} and outcome {outcome}')
	#
	# 	given = [self._mechanism_str_to_variable(term) for term in terms.split(';')]
	# 	return given


	# def symbolic_marginal_variables(self, outcome: str) -> List[str]:
	# 	terms = self.config.get('marginal_nec', None)
	# 	if terms is None or outcome != 'Y':
	# 		raise self._MissingEstimandError(f'marginal unknown for outcome {outcome}')
	#
	# 	given = [self._mechanism_str_to_variable(term) for term in terms.split(';')]
	# 	return given


	def _integrate_unobserved_variables(self, given: Sequence[Union[str, AbstractVariable]]):
		mechs = [self.get_variable(g) if isinstance(g, str) else g for g in given]
		priors = self.marginals()

		for mech in mechs:
			if any(p in self.unobserved_variables for p in mech.parents):
				yield mech.partial(**{v: priors[v] for v in self.unobserved_variables})
			else:
				yield mech


	def symbolic_mechanism_details(self, mechanisms: Sequence[str], *, keep_unobserved=False):
		'''removes unobserved parents'''
		given = [self.get_variable(m) if isinstance(m, str) else m for m in mechanisms]
		if keep_unobserved or self.unobserved_variables is None:
			return {str(mech): mech.param.tolist() for mech in given}
		fixed = {str(mech) : mech.param.tolist() for mech in self._integrate_unobserved_variables(given)}
		return fixed


	def verbalize_mechanism_details(self, labels: Dict[str, str], mechanisms: Sequence[str], *,
	                             keep_unobserved=False) -> str:
		given = [self.get_variable(m) if isinstance(m, str) else m for m in mechanisms]
		if self.unobserved_variables is not None and not keep_unobserved:
			given = self._integrate_unobserved_variables(given)
		return self.verbalize_description(labels, mechanisms=given)


	# def symbolic_ate_given_info(self, outcome: str, treatment: str, *, treated: bool = True):
	# 	given = self.symbolic_ate_variables(outcome, treatment)
	# 	if self.unobserved_variables is None:
	# 		return {str(self.get_variable(g)): self.get_variable(g).param.tolist() for g in given}
	#
	# 	fixed = {str(mech) : mech.param.tolist() for mech in self._integrate_unobserved_variables(given)}
	# 	return fixed


	# def symbolic_ett_given_info(self, outcome: str, treatment: str, *, treated: bool = True):
	# 	given = self.symbolic_ett_variables(outcome, treatment)
	# 	if self.unobserved_variables is None:
	# 		return {str(self.get_variable(g)): self.get_variable(g).param.tolist() for g in given}
	# 	return {str(mech) : mech.param.tolist() for mech in self._integrate_unobserved_variables(given)}


	# def symbolic_nde_given_info(self, outcome: str, treatment: str, *, mediators: Optional[Sequence[str]] = None,
	#                             treated: bool = True):
	# 	given = self.symbolic_nde_variables(outcome, treatment, mediators=mediators)
	# 	if self.unobserved_variables is None:
	# 		return {str(self.get_variable(g)): self.get_variable(g).param.tolist() for g in given}
	# 	return {str(mech) : mech.param.tolist() for mech in self._integrate_unobserved_variables(given)}


	# def symbolic_nie_given_info(self, outcome: str, treatment: str, *, mediators: Optional[Sequence[str]] = None,
	#                             treated: bool = True):
	# 	given = self.symbolic_nie_variables(outcome, treatment, mediators=mediators)
	# 	if self.unobserved_variables is None:
	# 		return {str(self.get_variable(g)): self.get_variable(g).param.tolist() for g in given}
	# 	return {str(mech) : mech.param.tolist() for mech in self._integrate_unobserved_variables(given)}


	# def verbalize_ate_given_info(self, labels: Dict[str, str], outcome: str, treatment: str) -> str:
	# 	if self.unobserved_variables is None:
	# 		return super().verbalize_ate_given_info(labels, outcome, treatment)
	# 	return self.verbalize_description(labels, mechanisms=self._integrate_unobserved_variables(
	# 		self.symbolic_ate_variables(outcome, treatment)))
	#
	#
	# def verbalize_ett_given_info(self, labels: Dict[str, str], outcome: str, treatment: str) -> str:
	# 	if self.unobserved_variables is None:
	# 		return super().verbalize_ett_given_info(labels, outcome, treatment)
	# 	return self.verbalize_description(labels, mechanisms=self._integrate_unobserved_variables(
	# 		self.symbolic_ett_variables(outcome, treatment)))
	#
	#
	# def verbalize_nde_given_info(self, labels: Dict[str, str], outcome: str, treatment: str) -> str:
	# 	if self.unobserved_variables is None:
	# 		return super().verbalize_nde_given_info(labels, outcome, treatment)
	# 	return self.verbalize_description(labels, mechanisms=self._integrate_unobserved_variables(
	# 		self.symbolic_nde_variables(outcome, treatment)))
	#
	#
	# def verbalize_nie_given_info(self, labels: Dict[str, str], outcome: str, treatment: str) -> str:
	# 	if self.unobserved_variables is None:
	# 		return super().verbalize_nie_given_info(labels, outcome, treatment)
	# 	return self.verbalize_description(labels, mechanisms=self._integrate_unobserved_variables(
	# 		self.symbolic_nie_variables(outcome, treatment)))
	#
	#
	# def verbalize_marginal_given_info(self, labels: Dict[str, str], outcome: str) -> str:
	# 	if self.unobserved_variables is None:
	# 		return super().verbalize_marginal_given_info(labels, outcome)
	# 	return self.verbalize_description(labels, mechanisms=self._integrate_unobserved_variables(
	# 		self.symbolic_marginal_variables(outcome)))



########################################################################################################################
# region Confounding



@register_graph('confounding')
class Confounding(Phenomenon):
	V1 = node()
	X = node(V1)
	Y = node(V1, X)



@register_graph('mediation')
class Mediation(Phenomenon):
	X = node()
	V2 = node(X)
	Y = node(X, V2)



@register_graph('triangle')
class Triangle(Phenomenon):
	X = node()
	Y = node(X)
	V3 = node(X, Y)



# endregion
########################################################################################################################
# region Two Causes



@register_graph('fork')
class Fork(Phenomenon):
	X = node()
	V2 = node()
	Y = node(X, V2)



@register_graph('collision')
class Collision(Phenomenon):
	X = node()
	Y = node()
	V3 = node(X, Y)



# endregion
########################################################################################################################



@register_graph('chain')
class Chain(Phenomenon):
	X = node()
	V2 = node(X)
	Y = node(V2)



@register_graph('IV')
class InstrumentalVariable(Phenomenon):
	unobserved_variables = hparam(['V1'])

	V1 = node()
	V2 = node()
	X = node(V1, V2)
	Y = node(V1, X)



# @register_scm('arrowheadcollision') # alias
@register_graph('arrowhead')
class Arrowhead(Phenomenon):
	unobserved_variables = hparam(['V2'])

	X = node()
	V2 = node()
	V3 = node(X, V2)
	Y = node(X, V2, V3)
	


@register_graph('frontdoor')
class Frontdoor(Phenomenon):
	unobserved_variables = hparam(['V1'])

	V1 = node()
	X = node(V1)
	V3 = node(X)
	Y = node(V1, V3)



########################################################################################################################
# region Diamond



@register_graph('diamond')
class Diamond(Phenomenon):
	X = node()
	V2 = node(X)
	V3 = node(X)
	Y = node(V2, V3)



@register_graph('diamondcut')
class DiamondCut(Phenomenon):
	V1 = node()
	X = node(V1)
	V3 = node(V1)
	Y = node(X, V3)


# endregion






