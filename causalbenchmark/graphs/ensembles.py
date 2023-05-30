from typing import Union, Dict, List, Tuple, Iterator, Optional, Any, Iterable, Callable, Sequence, Set, TypeVar, Type, cast
import numpy as np
from itertools import product

from omnibelt import unspecified_argument
from omniply import Parameterized, hparam, submodule

from scipy.linalg import pascal

from ..util import Seeded

from .base import get_graph_class, create_graph
from .bayes import BayesLike, CausalProcess
# from .builders import RelativeBuilder



class SCMEnsemble(CausalProcess, Parameterized):
	'''Uses Monte Carlo sampling to estimate the bounds of a distribution of SCMs.'''
	story = hparam(required=True, inherit=True)
	spec = hparam(required=True, inherit=True)
	builder = hparam(required=True, inherit=True)

	size = hparam(default=5, inherit=True)


	def verbalize_graph(self, labels: Dict[str, str]) -> str:
		return self.ensemble[0].verbalize_graph(labels)


	def verbalize_mechanisms(self, labels: Dict[str, str]) -> str:
		return self.builder.verbalize_spec(labels, story=self.story, spec=self.spec)


	def __init__(self, ensemble=None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self.builder.is_deterministic:
			print(f'WARNING: setting num_samples to 1 for deterministic builder {self.builder} '
			      f'(instead of {self.size})')
			self.size = 1
		if ensemble is None:
			ensemble = self._build_ensemble()
		self.ensemble = ensemble


	def copy(self, graph_id=unspecified_argument, spec=unspecified_argument,
	         builder=unspecified_argument, ensemble=unspecified_argument, **kwargs):
		if graph_id is unspecified_argument:
			graph_id = self.graph_id
		if spec is unspecified_argument:
			spec = self.spec
		if builder is unspecified_argument:
			builder = self.builder
		if ensemble is unspecified_argument:
			ensemble = self.ensemble.copy()
		return self.__class__(graph_id=graph_id, spec=spec, builder=builder, ensemble=ensemble, **kwargs)


	def intervene(self, **interventions) -> 'SCMEnsemble': # TODO: maybe remove this
		return self.copy(ensemble=[sample.intervene(**interventions) for sample in self.ensemble])


	def _build_ensemble(self, **sample_kwargs):
		ensemble = []
		for _ in range(self.size):
			ensemble.append(self.builder.generate_scm_example(self.story, spec=self.spec, **sample_kwargs))
		return ensemble


	def __len__(self):
		return self.size


	def __iter__(self):
		yield from self.ensemble


	@staticmethod
	def _extract_bounds(itr: Iterator[Dict[str, float]]) -> Dict[str, List[float]]:
		lb = {}
		ub = {}

		for individual in itr:
			for k, v in individual.items():
				if k not in lb:
					lb[k] = v
					ub[k] = v
				else:
					lb[k] = min(lb[k], v)
					ub[k] = max(ub[k], v)

		return {k: [lb[k], ub[k]] for k in lb}


	def full_ate_bounds(self, treatment: str, *, treated: bool = True) -> Dict[str, List[float]]:
		return self._extract_bounds(self.ate_samples(treatment, treated=treated))


	def full_ett_bounds(self, treatment: str, *, treated: bool = True) -> Dict[str, List[float]]:
		return self._extract_bounds(self.ett_samples(treatment, treated=treated))


	def full_nde_bounds(self, treatment: str, outcome: str, *, mediators: Optional[Sequence[str]] = None,
	                    treated: bool = True) -> Dict[str, List[float]]:
		return self._extract_bounds(self.nde_samples(treatment, outcome, mediators=mediators, treated=treated))


	def full_nie_bounds(self, treatment: str, outcome: str, *, mediators: Optional[Sequence[str]] = None,
	                    treated: bool = True) -> Dict[str, List[float]]:
		return self._extract_bounds(self.nie_samples(treatment, outcome, mediators=mediators, treated=treated))


	def full_marginal_bounds(self, **conditions: list) -> Dict[str, List[float]]:
		return self._extract_bounds(self.marginal_samples(**conditions))


	def full_interventional_bounds(self, interventions: Dict[str, float], *,
	                          conditions: Optional[Dict[str, int]] = None) -> Dict[str, List[float]]:
		return self._extract_bounds(self.interventional_samples(interventions, conditions=conditions))


	def full_counterfactual_bounds(self, *, factual: Optional[Dict[str, int]] = None,
	                          evidence: Optional[Dict[str, int]] = None,
	                          action: Optional[Dict[str, float]] = None) -> Dict[str, List[float]]:
		return self._extract_bounds(self.counterfactual_samples(factual=factual, evidence=evidence, action=action))


	def ate_samples(self, treatment: str, *, treated: bool = True) -> Iterator[Dict[str, float]]:
		for sample in self:
			yield sample.ate(treatment, treated=treated)


	def ett_samples(self, treatment: str, *, treated: bool = True) -> Iterator[Dict[str, float]]:
		for sample in self:
			yield sample.ett(treatment, treated=treated)


	def nde_samples(self, treatment: str, outcome: str, *, mediators: Optional[Sequence[str]] = None,
	                treated: bool = True) -> Iterator[Dict[str, float]]:
		for sample in self:
			yield sample.nde(treatment, outcome, mediators=mediators, treated=treated)


	def nie_samples(self, treatment: str, outcome: str, *, mediators: Optional[Sequence[str]] = None,
	                treated: bool = True) -> Iterator[Dict[str, float]]:
		for sample in self:
			yield sample.nie(treatment, outcome, mediators=mediators, treated=treated)


	def marginal_samples(self, **conditions): # var={0,1}
		for sample in self:
			yield sample.marginals(**conditions)


	def interventional_samples(self, interventions: Dict[str, float], *,
	                          conditions: Optional[Dict[str, int]] = None) -> Iterator[Dict[str, float]]:
		if conditions is None:
			conditions = {}
		for sample in self:
			yield sample.intervene(**interventions).marginals(**conditions)


	def counterfactual_samples(self, *, factual: Optional[Dict[str, int]] = None,
	                           evidence: Optional[Dict[str, int]] = None,
	                           action: Optional[Dict[str, float]] = None) -> Iterator[Dict[str, float]]:
		for sample in self:
			yield sample.counterfactual(factual=factual, evidence=evidence, action=action)



########################################################################################################################



# class RelativeSCMEnsemble(SCMEnsemble):
# 	_correlation_template = 'The chance of {outcome} {change} when {cause} compared to {baseline}.'
# 	_source_template = 'The chance of {variable} is {prob} {alternative}.'
#
# 	_incdec = ['increases', 'decreases']
#
#
# 	num_samples = hparam(5, inherit=True)
#
#
# 	bin_labels = hparam(['much lower than', 'about the same as', 'much higher than'])
#
#
# 	def __len__(self):
# 		return len(self.ensemble)
#
#
# 	def __iter__(self):
# 		yield from self.ensemble
#
#
# 	def description(self, labels):
# 		return self.ensemble[0].description(labels)
#
#
# 	def details(self, labels): # verbalizes the details of the story mechanisms
#
# 		spec = self.spec
#
# 		lines = []
#
# 		for node in self.nodes():
# 			info = spec[node.name]
# 			if len(node.parents) == 0:
# 				lines.append(self._source_template.format(variable=labels[f'{node.name}1'],
# 				                                    prob=self.bin_labels[info],
# 				                                    alternative=labels[f'{node.name}0'],
# 				                                    **labels))
# 			else:
# 				outcome = labels[f'{node.name}1']
# 				for parent in node.parents:
# 					if parent not in spec or spec[parent] is None or spec[parent] == 0:
# 						continue
#
# 					cause = labels[f'{parent}1']
# 					baseline = labels[f'{parent}0']
# 					change = self._incdec[int(info[parent] < 0)]
#
# 					lines.append(self._correlation_template.format(outcome=outcome, cause=cause, change=change,
# 					                                               baseline=baseline, **labels))
#
# 		return ' '.join(lines)
#
#
# 	def __init__(self, builder=None, ensemble=None, *args, **kwargs):
# 		if builder is None:
# 			builder = RelativeBuilder()
# 		super().__init__(*args, **kwargs)
# 		self.builder = builder
# 		self.ensemble = ensemble
# 		if self.builder.is_deterministic:
# 			print(f'WARNING: setting num_samples to 1 for deterministic builder {self.builder} '
# 			      f'(instead of {self.num_samples})')
# 			self.num_samples = 1
# 		if ensemble is None:
# 			ensemble = self._build_ensemble()
# 		self.ensemble = ensemble
#
#
# 	def copy(self, builder=unspecified_argument, ensemble=unspecified_argument, **kwargs):
# 		if builder is unspecified_argument:
# 			builder = self.builder
# 		if ensemble is unspecified_argument:
# 			ensemble = self.ensemble.copy()
# 		return self.__class__(builder=builder, ensemble=ensemble, **kwargs)
#
#
# 	def _build_ensemble(self, **sample_kwargs):
# 		ensemble = []
# 		for _ in range(self.num_samples):
# 			ensemble.append(self.builder.create(self.graph_id, spec=self.spec, **sample_kwargs))
# 		return ensemble
#
#
	def _unordered_nodes(self):
		yield from self.ensemble[0].nodes()
#
#
# 	def intervene(self, **interventions) -> 'BayesLike': # TODO: maybe remove this
# 		return self.copy(ensemble=[sample.intervene(**interventions) for sample in self.ensemble])






