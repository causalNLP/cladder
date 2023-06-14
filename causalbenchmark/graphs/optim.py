from typing import Optional, Any, Iterator, Hashable, Type, Union, List, Dict, Tuple, Sequence, Callable, Generator
import numpy as np
from scipy import optimize as opt
from scipy import special

from .base import create_graph


class ProbabilityConstraint:
	def __init__(self, spec: str, bounds: Union[float, Tuple[float, float]] = None, *,
	             wt: float = 1.0, p: float = 2., target: Optional[float] = None, eps: Optional[float] = 0.01):
		if bounds is None:
			assert target is not None, 'Must specify either bounds or target'
			bounds = target, target
		if isinstance(bounds, float):
			target = bounds
			bounds = target, target
		lb, ub = bounds
		lb = max(lb-eps, 0)
		ub = min(ub+eps, 1)
		bounds = lb, ub
		if target is None:
			target = (lb + ub) / 2
		self.spec = spec
		self.name, self.given_spec = self.parse_spec(spec)
		self.given = self.parse_given_spec(self.given_spec)
		self.bounds = bounds
		self.target = target
		self.wt = wt
		self.p = p


	def as_constraint(self, graph, **kwargs):
		return self._Constraint(self, graph, **kwargs)


	class _Constraint(opt.NonlinearConstraint):
		def __init__(self, owner, graph, **kwargs):
			super().__init__(self._func, *owner.bounds, **kwargs)
			self.owner = owner
			self.graph = graph


		def _func(self, x):
			self.graph.set_parameters(x)
			value = self.owner.estimate(self.graph)
			return value


	@staticmethod
	def parse_spec(spec: str):
		name, *given_spec = spec.strip().split('|')
		given_spec = given_spec[0] if len(given_spec) else ''
		return name.strip(), given_spec.strip()


	@staticmethod
	def parse_given_spec(given_spec: str):
		if len(given_spec):
			given = {k.strip(): int(v.strip())
			         for k, v in (x.strip().split('=') for x in given_spec.strip().split(','))}
		else:
			given = None
		return given


	def compute_loss(self, model, cache=None):
		value = self.estimate(model, cache)
		loss = np.abs(value - self.target) ** self.p
		return loss * self.wt


	def estimate(self, model, cache=None):
		if self.given is None:
			if cache is None:
				cache = model.marginals()
			elif self.name not in cache:
				cache.update(model.marginals())
			return cache[self.name]

		if cache is None:
			cache = model.marginals(**self.given)
		elif self.given_spec not in cache:
			cache[self.given_spec] = model.marginals(**self.given)
			cache = cache[self.given_spec]
		return cache[self.name]



def optim_graph(graph, constraints, *, decay=0.01, eps=0.001):
	x0 = graph.get_parameters()

	def step(x):
		graph.set_parameters(x)
		cache = {}
		return sum(c.compute_loss(graph, cache) for c in constraints) + decay * np.sum((x - x0) ** 2)

	prob_constraint = opt.LinearConstraint(np.eye(len(x0)), eps, 1 - eps, keep_feasible=True)

	sol = opt.minimize(step, x0, constraints=[prob_constraint])

	graph.set_parameters(sol.x)
	return graph



def test_optim_params():

	graph = create_graph('confounding')

	targets = {
		'X': 0.5,
		'V1': 0.5,
		'Y|X=0': 0.5,
		'Y|X=1,Z=1': 0.5,
	}

	constraints = [ProbabilityConstraint(spec, target) for spec, target in targets.items()]

	p0 = graph.get_parameters()

	graph = optim_graph(graph, constraints)

	p1 = graph.get_parameters()

	result = \
		{constraint.spec: constraint.estimate(graph) for constraint in constraints}

	print(targets)
	print(result)



def test_constrained_ate():

	graph_id = 'confounding'

	targets = {
		'X': [0.3, 0.5],
		'V1': [0.5, 0.8],
		'Y|X=0': [0., 0.5],
		'Y|X=1,Z=1': [0.5, 1.],
	}

	constraints = [ProbabilityConstraint(spec, target) for spec, target in targets.items()]

	graph = create_graph(graph_id)
	graph = optim_graph(graph, constraints)

	x0 = graph.get_parameters()

	initial = \
		{constraint.spec: constraint.estimate(graph) for constraint in constraints}

	optim_constraints = [c.as_constraint(graph) for c in constraints]

	eps = 0.001

	prob_constraint = opt.LinearConstraint(np.eye(len(x0)), eps, 1 - eps, keep_feasible=True)

	treatment, outcome = 'X', 'Y'
	treated = True

	subject = create_graph(graph_id)
	def ate_objective(params):
		# params = params.clip(0, 1)
		model = subject.set_parameters(params)
		ate = model.ate(treatment, treated=treated)[outcome]
		return ate

	objective = ate_objective
	sol = opt.minimize(objective, x0, constraints=[prob_constraint, *optim_constraints])
	nsol = opt.minimize(lambda x: -objective(x), x0, constraints=[prob_constraint, *optim_constraints])

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


	print(f'{lb:.3f} <= ATE <= {ub:.3f}')
	assert -1 <= lb <= ub <= 1, 'ATE out of bounds'




