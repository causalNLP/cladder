from typing import Union, Dict, List, Tuple, Iterator, Optional, Any, Iterable, Callable, Sequence, Set, TypeVar, Type, cast
import numpy as np
from itertools import product

from omnibelt import unspecified_argument, JSONABLE
from omniply import Parameterized, hparam, submodule

from scipy.linalg import pascal
from scipy import optimize as opt

from ..util import Seeded

from .base import get_graph_class, create_graph, AbstractVariable, AbstractProcess
from .bayes import BayesLike, CausalProcess



class ConstrainedSCM(CausalProcess, Seeded):
	graph_id = hparam(required=True, inherit=True)
	spec = hparam(required=True, inherit=True)
	builder = hparam(required=True, inherit=True)

	method = hparam(None, inherit=True)

	describe_all_constraints = hparam(True, inherit=True) # stylistic choice


	def verbalize_background(self, labels: Dict[str, str]) -> str:
		return self.subject.verbalize_background(labels)


	def verbalize_graph(self, labels: Dict[str, str]) -> str:
		return self.subject.verbalize_graph(labels)


	def verbalize_description(self, labels: Dict[str, str],
	                          mechanisms: Optional[Sequence[Union[str, AbstractVariable]]] = None) -> str:
		spec = self.spec
		if mechanisms is not None:
			spec = {k: v for k, v in spec.items() if k in mechanisms}
		return self.builder.verbalize_spec(labels, graph_id, spec=spec)


	def verbalize_ate_given_info(self, labels: Dict[str, str], outcome: str, treatment: str) -> str:
		if self.describe_all_constraints:
			return self.verbalize_description(labels)
		return super().verbalize_ate_given_info(labels, outcome, treatment)


	def symbolic_ate_given_info(self, outcome: str, treatment: str, *, treated: bool = True) -> Dict[str, JSONABLE]:
		return {}


	def verbalize_ett_given_info(self, labels: Dict[str, str], outcome: str, treatment: str) -> str:
		if self.describe_all_constraints:
			return self.verbalize_description(labels)
		return super().verbalize_ett_given_info(labels, outcome, treatment)


	def symbolic_ett_given_info(self, outcome: str, treatment: str, *, treated: bool = True) -> Dict[str, JSONABLE]:
		return {}


	subject = None
	def _create_subject(self):
		if self.subject is None:
			return create_graph(self.graph_id)
		return self.subject


	def _initial_parameters(self):
		# return self.builder.generate_scm_example(self.graph_id, self.spec).get_parameters()
		# x0 = self._rng.uniform(0, 1, self.subject.dof())
		x0 = np.ones(self.subject.dof()) * 0.5
		return x0


	def __init__(self, constraints=None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.constraints = constraints
		self.subject = self._create_subject()


	def copy(self, graph_id=unspecified_argument, spec=unspecified_argument,
	         constraints=unspecified_argument, **kwargs):
		if graph_id is unspecified_argument:
			graph_id = self.graph_id
		if spec is unspecified_argument:
			spec = self.spec
		if constraints is unspecified_argument:
			constraints = self.constraints
		return self.__class__(graph_id=graph_id, spec=spec, constraints=constraints, **kwargs)


	def unordered_variables(self) -> Iterator[AbstractVariable]:
		yield from self.subject.unordered_variables()


	def find_bounds(self, objective, x0=None) -> List[float]:
		if x0 is None:
			x0 = self._initial_parameters()

		sol = opt.minimize(objective, x0, constraints=self.constraints, method=self.method)
		nsol = opt.minimize(lambda x: -objective(x), x0, constraints=self.constraints, method=self.method)

		if not sol.success or not nsol.success:
			print('WARNING: optimization failed')
			print(sol)
			print(nsol)

		lb = sol.fun
		if isinstance(lb, np.ndarray):
			lb = lb.item()

		ub = -nsol.fun
		if isinstance(ub, np.ndarray):
			ub = ub.item()

		return [lb, ub]


	def make_ate_objective(self, outcome: str, treatment: str, *, treated: bool = True):
		def ate_objective(params):
			# params = params.clip(0, 1)
			model = self.subject.set_parameters(params)
			ate = model.ate(treatment, treated=treated)[outcome]
			return ate
		return ate_objective


	def make_ett_objective(self, outcome: str, treatment: str, *, treated: bool = True):
		def ett_objective(params):
			return self.subject.set_parameters(params).ett(treatment, treated=treated)[outcome]
		return ett_objective


	def make_nde_objective(self, outcome: str, treatment: str, *mediators: str, treated: bool = True):
		def nde_objective(params):
			return self.subject.set_parameters(params).nde(treatment, *mediators, treated=treated)[outcome]
		return nde_objective


	def make_nie_objective(self, outcome: str, treatment: str, *mediators: str, treated: bool = True):
		def nie_objective(params):
			return self.subject.set_parameters(params).nie(treatment, *mediators, treated=treated)[outcome]
		return nie_objective


	def make_marginal_objective(self, outcome: str, **conditions: str):
		def marginal_objective(params):
			return self.subject.set_parameters(params).marginals(**conditions)[outcome]
		return marginal_objective


	def make_interventional_objective(self, outcome: str, interventions: Dict[str, float], *,
	                                  conditions: Optional[Dict[str, int]] = None):
		def interventional_objective(params):
			return self.subject.set_parameters(params).interventional(interventions, conditions=conditions)[outcome]
		return interventional_objective


	def make_counterfactual_objective(self, outcome: str, *, factual: Optional[Dict[str, int]] = None,
	                                  evidence: Optional[Dict[str, int]] = None,
	                                  action: Optional[Dict[str, float]] = None):
		def counterfactual_objective(params):
			return self.subject.set_parameters(params).counterfactual(factual=factual, evidence=evidence,
			                                                          action=action)[outcome]
		return counterfactual_objective


	def ate_bounds(self, outcome: str, treatment: str, *, treated: bool = True) -> List[float]:
		return self.find_bounds(self.make_ate_objective(outcome, treatment, treated=treated))


	def ett_bounds(self, outcome: str, treatment: str, *, treated: bool = True) -> List[float]:
		return self.find_bounds(self.make_ett_objective(outcome, treatment, treated=treated))


	def nde_bounds(self, outcome: str, treatment: str, *mediators: str, treated: bool = True) -> List[float]:
		return self.find_bounds(self.make_nde_objective(outcome, treatment, *mediators, treated=treated))


	def nie_bounds(self, outcome: str, treatment: str, *mediators: str, treated: bool = True) -> List[float]:
		return self.find_bounds(self.make_nie_objective(outcome, treatment, *mediators, treated=treated))


	def marginal_bounds(self, outcome: str, **conditions: str) -> List[float]:
		return self.find_bounds(self.make_marginal_objective(outcome, **conditions))


	def interventional_bounds(self, outcome: str, interventions: Dict[str, float], *,
	                          conditions: Optional[Dict[str, int]] = None) -> List[float]:
		return self.find_bounds(self.make_interventional_objective(outcome, interventions, conditions=conditions))


	def counterfactual_bounds(self, outcome: str, *, factual: Optional[Dict[str, int]] = None,
	                          evidence: Optional[Dict[str, int]] = None,
	                          action: Optional[Dict[str, float]] = None) -> List[float]:
		return self.find_bounds(self.make_counterfactual_objective(outcome, factual=factual, evidence=evidence,
		                                                           action=action))









