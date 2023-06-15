from typing import Type, Any, Dict, Union, Iterator, Optional, Callable, Tuple, List, Sequence, Iterable, Set, Mapping
import numpy as np
from omniply import Structured
import pomegranate as pg
from functools import lru_cache

from omnibelt import agnostic, unspecified_argument, JSONABLE

from omniply import Context
from omniply.tools.crafts import Signatured, ToolCraft, AbstractTool
from omniply.tools.kits import DynamicKit

from .. import util
from ..util import Seeded, hparam

from .base import AbstractVariable, AbstractProcess, CausalProcess, SCM



class AbstractBernoulliMechanism(AbstractVariable):
	def dof(self) -> int:
		return 2**len(self.parents)


	def generate_parameters(self, seed=None, *, gen=None):
		if gen is None:
			gen = np.random.default_rng(seed)
		return gen.random(2**len(self.parents)).reshape([2]*len(self.parents))


	def format_parameters(self, val=None):
		if val is None:
			val = self.generate_parameters()
		else:
			val = np.asarray(val)
			assert val.size == 2**len(self.parents)
			val = val.reshape(*[2]*len(self.parents)) if len(self.parents) else np.array(val)
			# assert val.min() >= 0 and val.max() <= 1, f'Parameters must be in [0, 1]: {val}'
		return val


	def prob(self, val: float = 1, **conds: float):
		'''Conds must include all parents!'''
		assert len(conds) == len(self.parents), f'Must condition on all parents: {self.parents} exactly'

		total = self.partial(**conds).param.item()
		return val * total + (1-val) * (1 - total)


	def _create_mechanism(self, name, parents, param):
		raise NotImplementedError


	def partial(self, **conds: float) -> 'AbstractBernoulliMechanism':
		vals = self.param
		for i, parent in enumerate(self.parents):
			if parent in conds:
				x = np.array([1-conds[parent], conds[parent]]).reshape([1]*i + [2] + [1]*(len(self.parents)-i-1))
				vals = vals * x
				vals = vals.sum(axis=i, keepdims=True)

		vals = vals.squeeze()
		return self._create_mechanism(self.name, [p for p in self.parents if p not in conds], vals)


	def parameter_index(self, **conds: int) -> int:
		assert len(conds) == len(self.parents), f'Must condition on all parents: {self.parents} exactly (got {conds})'
		if len(conds) == 0:
			return 0

		binary = ''.join(str(conds[parent]) for parent in self.parents)
		return int(binary, 2)


	@staticmethod
	def _bernoulli_prior(p, N):
		return (np.random.rand(N) < p).astype(int)


	@staticmethod
	def _bernoulli_mechanism(params, *parents):
		# print(parents)
		if isinstance(parents[0], int):
			return (np.random.rand() < params[parents]).astype(int)
		if not len(parents):
			raise ValueError('Bernoulli mechanism must have at least one parent (otherwise use _bernoulli_prior)')
		return (np.random.rand(parents[0].size) < params[parents]).astype(int)


	def __str__(self):
		return f'p({self.name} | {", ".join(self.parents)})' if len(self.parents) else f'p({self.name})'


	def __repr__(self):
		arg = f'{self.name} | {", ".join(self.parents)}' if len(self.parents) else f'{self.name}'
		return f'{self.__class__.__name__}({arg})'



class BernoulliMechanism(AbstractBernoulliMechanism, AbstractTool):
	_params = None
	def __init__(self, *parents: Union['node', str], name: Optional[str] = None, param = None, **kwargs):
		super().__init__(**kwargs)
		self._name = name
		self._parents = parents
		self.param = param


	def _create_mechanism(self, name, parents, param):
		return self.__class__(*parents, name=name, param=param)


	def has_gizmo(self, gizmo: str) -> bool:
		return gizmo == self.name


	def gizmos(self):
		yield self.name


	def get_from(self, ctx: 'BayesNetContext', gizmo: str):
		assert gizmo == self.name, f'Expected {self.name}, got {gizmo}'
		if not len(self.parents):
			return self._bernoulli_prior(self.param, ctx.size)

		parents = [ctx[parent] for parent in self.parents]
		return self._bernoulli_mechanism(self.param, *parents)


	@property
	def name(self):
		return self._name


	@property
	def parents(self):
		return self._parents


	@property
	def param(self):
		if self._params is None:
			self.param = None
		return self._params
	@param.setter
	def param(self, val):
		self._params = self.format_parameters(val)



class node_(BernoulliMechanism, ToolCraft.Skill):
	_base: 'node'
	def __init__(self, *, parents: Optional[Sequence[str]] = None,
	             name: Optional[str] = None, param = None, **kwargs):
		super().__init__(name=name, param=param, **kwargs)
		self._parents = parents


	def _create_mechanism(self, name, parents, param):
		return BernoulliMechanism(*parents, name=name, param=param)


	@property
	def name(self):
		if self._name is None:
			return self._base.name
		return self._name


	@property
	def parents(self):
		if self._parents is None:
			return self._base.parents
		return self._parents



class node(AbstractBernoulliMechanism, ToolCraft):
	def __init__(self, *parents: Union['node', str], label: Optional[str] = None, p = None, **kwargs):
		super().__init__(label=label, **kwargs)
		self._parents = parents
		self._default_params = p


	@property
	def name(self):
		return self._label


	@property
	def parents(self):
		return tuple(parent.label if isinstance(parent, node) else parent for parent in self._parents)


	@property
	def default_param(self):
		return self._default_params
	@default_param.setter
	def default_param(self, val):
		self._default_params = val


	# region Craft Responsibilities

	def __set_name__(self, owner, name):
		self._label = name


	Skill = node_
	def as_skill(self, instance):
		skill = self.Skill(parents=self.parents, name=self.name, param=self.default_param,
		                   base=self, instance=instance)
		setattr(instance, self.label, skill)
		return skill


	def replace(self, **kwargs):
		raise NotImplementedError


	def emit_craft_items(self, owner=None): # avoid dealing with "wrapped"
		yield self


	# def _get_from(self, instance, ctx, gizmo: str):
	# 	return getattr(instance, self.label).get_from(ctx, gizmo)


	# def __get__(self, instance, owner):
	# 	return self

	# def _unordered_nodes(self):
	# 	yield self

	# endregion
	pass



########################################################################################################################



class BayesLike(CausalProcess, AbstractTool):
	# region Pomgranate
	@staticmethod
	def _as_pom_node(x, pool):
		if len(x.parents):
			parents = [pool[p] for p in x.parents]
			table = [[*inds, x, p if x else 1-p]
			         for inds, p in zip(np.ndindex(*x.param.shape), x.param.flat)
			         for x in [0, 1]]

			return pg.ConditionalProbabilityTable(table, parents)
		return pg.DiscreteDistribution({0: 1 - x.param, 1: x.param})


	def as_pom(self, name='model', **kwargs):
		graph = {}
		pool = {}

		for x in self.variables():
			graph[x.name] = x.parents
			pool[x.name] = self._as_pom_node(x, pool)

		nodes = {name: pg.Node(node, name=name) for name, node in pool.items()}

		net = pg.BayesianNetwork(name, **kwargs)
		net.add_states(*nodes.values())

		for name, parents in graph.items():
			for parent in parents:
				net.add_edge(nodes[parent], nodes[name])

		net.bake()
		return net
	# endregion


	def covariance(self, var1: str, var2: str, **conditions: int) -> float:
		# marginals = self.marginals(**conditions)
		# return self.probability(**{var1: 1, var2: 1}, **conditions) - marginals[var1] * marginals[var2]
		return (self.marginals(**{var2: 1}, **conditions)[var1] - self.marginals(**{var2: 0}, **conditions)[var1]) \
			* self.variances(**conditions)[var2]


	def variances(self, **conditions: int) -> Dict[str, float]:
		return {var: p*(1-p) for var, p in self.marginals(**conditions).items()}


	def correlation(self, var1: str, var2: str, **conditions: int) -> float:
		sigmas = self.variances(**conditions)
		return self.covariance(var1, var2, **conditions) / np.sqrt(sigmas[var1] * sigmas[var2])


	def probability(self, **evidence: int) -> float: # TODO: not super efficient :/
		vs = list(self.variable_names())
		full = util.generate_all_bit_strings(len(vs)).reshape(*[2] * len(vs), len(vs))

		sel = [(evidence[v] if v in evidence else slice(None)) for v in vs]

		rows = full[tuple(sel)].reshape(-1, len(vs))

		net = self.as_pom()
		return net.probability(rows).sum().item()


	def distribution(self, variable, *parents: str, **conditions: int) -> AbstractVariable:
		params = []

		for combo in util.generate_all_bit_strings(len(parents)):
			conditioning = dict(zip(parents, combo))
			# params.append(self.probability(**{variable: 1}, **conditioning, **conditions))
			params.append(self.marginals(**conditioning, **conditions)[variable])


		return BernoulliMechanism(*parents, name=variable, param=params)


	# region Direct Stats
	def marginals(self, **conditions: int) -> Dict[str, float]:
		net = self.as_pom()
		raw = net.predict_proba(conditions)
		return {state.name: term if state.name in conditions else float(term.parameters[0][1])
		        for state, term in zip(net.states, raw)}

	def interventional(self, interventions: Dict[str, float],
	                          conditions: Optional[Dict[str, int]] = None) -> Dict[str, float]:
		if conditions is None:
			conditions = {}
		return self.intervene(**interventions).marginals(**conditions)


	def counterfactual(self, *, factual: Optional[Dict[str, int]] = None, evidence: Optional[Dict[str, int]] = None,
	                   action: Optional[Dict[str, float]] = None) -> Dict[str, float]:
		'''
		Note that while the values of factual and evidence are binary,
		the values of action are continuous (-> soft interventions)
		'''
		if factual is None:
			factual = {}
		if action is None:
			action = {}
		if evidence is None:
			evidence = {}

		abduction = self.marginals(**factual, **evidence)
		sources = {src.name: abduction[src.name] for src in self.source_variables() if src.name not in action}
		return self.interventional({**action, **sources}, evidence)


	def ate(self, treatment: str, *, treated: bool = True):
		'''Average causal effect of `treatment` on all variables'''
		do_1 = self.interventional({treatment: int(treated)})
		do_0 = self.interventional({treatment: int(not treated)})
		return {name: do_1[name] - do_0[name] for name in do_1}


	def ett(self, treatment: str, *, treated: bool = True):
		'''Effect of treatment on the treated on all variables'''
		obs = self.marginals(**{treatment: int(treated)})
		counterfactual = self.counterfactual(factual={treatment: int(treated)},
		                                     action={treatment: int(not treated)})
		return {name: obs[name] - counterfactual[name] for name in obs}


	def find_mediators(self, start: str, end: str, *, _not_direct=False):
		'''assumes graph is a DAG (otherwise, this is an infinite loop)'''
		for candidate in self.get_variable(end).parents:
			if candidate == start: # base case: parent is the start
				if _not_direct:
					yield end # return mediator if not direct
			else:
				# check parents of candidate recursively to find the start
				path = self.find_mediators(start, candidate, _not_direct=True)
				for mediator in path: # if a path is found
					yield mediator # return the mediator
					yield from path # expend the generator
					if mediator != candidate:
						yield candidate # include the successful candidate if not already included


	def find_colliders(self, start: str, end: str) -> Iterator[str]:
		for v in self.variables():
			if start in v.parents and end in v.parents:
				yield v.name


	def nde(self, treatment: str, outcome: str, *,
	        mediators: Optional[Sequence[str]] = None, treated: bool = True) -> Dict[str, float]:
		'''Natural direct effect `treatment` on all variables (wrt `mediators`)'''
		if mediators is None:
			mediators = list(self.find_mediators(treatment, outcome))
		if not mediators:
			return self.ate(treatment)

		do_x0 = self.interventional({treatment: int(not treated)})
		mx0 = {mediator: do_x0[mediator] for mediator in mediators}

		do_x1mx0 = self.interventional({treatment: int(treated), **mx0})

		nde = {name: do_x1mx0[name] - do_x0[name] for name in do_x0}
		return nde


	def nie(self, treatment: str, outcome: str, *,
	        mediators: Optional[Sequence[str]] = None, treated: bool = True) -> Dict[str, float]:
		'''Natural indirect effect of `treatment` on all variables (wrt `mediators`)'''
		if mediators is None:
			mediators = list(self.find_mediators(treatment, outcome))

		do_x1 = self.interventional({treatment: int(treated)})
		mx1 = {mediator: do_x1[mediator] for mediator in mediators}

		do_x0mx1 = self.interventional({treatment: int(not treated), **mx1})

		do_x0 = self.interventional({treatment: int(not treated)})

		return {name: do_x0mx1[name] - do_x0[name] for name in do_x0}
	# endregion


	# region Full Stats Interface
	def full_ate_bounds(self, treatment: str, *, treated: bool = True) -> Dict[str, List[float]]:
		ate = self.ate(treatment, treated=treated)
		return {name: [ate[name], ate[name]] for name in ate}

	def full_ett_bounds(self, treatment: str, *, treated: bool = True) -> Dict[str, List[float]]:
		ett = self.ett(treatment, treated=treated)
		return {name: [ett[name], ett[name]] for name in ett}

	def full_nde_bounds(self, treatment: str, outcome: str, *, mediators: Optional[Sequence[str]] = None,
	                    treated: bool = True) -> Dict[str, List[float]]:
		nde = self.nde(treatment, outcome, mediators=mediators, treated=treated)
		return {name: [nde[name], nde[name]] for name in nde}

	def full_nie_bounds(self, treatment: str, outcome: str, *, mediators: Optional[Sequence[str]] = None,
	                    treated: bool = True) -> Dict[str, List[float]]:
		nie = self.nie(treatment, outcome, mediators=mediators, treated=treated)
		return {name: [nie[name], nie[name]] for name in nie}

	def full_marginal_bounds(self, **conditions) -> Dict[str, List[float]]:
		obs = self.marginals(**conditions)
		return {name: [obs[name], obs[name]] for name in obs}

	def full_interventional_bounds(self, interventions: Dict[str, float], *,
	                               conditions: Optional[Dict[str, int]] = None) -> Dict[str, List[float]]:
		obs = self.interventional(interventions, conditions)
		return {name: [obs[name], obs[name]] for name in obs}

	def full_counterfactual_bounds(self, factual: Optional[Dict[str, int]] = None,
	                               evidence: Optional[Dict[str, int]] = None,
	                               action: Optional[Dict[str, float]] = None) -> Dict[str, List[float]]:
		obs = self.counterfactual(factual=factual, evidence=evidence, action=action)
		return {name: [obs[name], obs[name]] for name in obs}
	# endregion


	def intervene(self, **interventions: float) -> 'BayesLike': # var=[0,1]
		'''
		returns a new causal process where the interventions are applied
		soft interventions allowed
		'''
		return self.context().include(IndependentBernoullis(**interventions))


	def context(self, size=None, **kwargs):
		'''behaves like a dictionary'''
		return BayesNetContext(size=size, **kwargs).include(self)


	def sample(self, N=None):
		'''lazy sampling, returns a context that can be used like a dictionary to lazily sample from the process'''
		return self.context(size=N)


	def unordered_variables(self) -> Iterator[AbstractBernoulliMechanism]:
		past = set()
		for tool in self.tools():
			if isinstance(tool, AbstractVariable):
				if tool.name not in past:
					yield tool
					past.add(tool.name)
			elif isinstance(tool, AbstractProcess):
				for var in tool.unordered_variables():
					if var.name not in past:
						yield var
						past.add(var.name)


	def get_variable(self, name: str, default=unspecified_argument):
		for tool in self.tools():
			if isinstance(tool, AbstractVariable):
				if tool.name == name:
					return tool
			elif isinstance(tool, AbstractProcess):
				try:
					return tool.get_variable(name)
				except AttributeError:
					pass
		raise AttributeError(name)


	def has_variable(self, name: str) -> bool:
		for tool in self.tools():
			if isinstance(tool, AbstractVariable):
				if tool.name == name:
					return True
			elif isinstance(tool, AbstractProcess):
				if tool.has_variable(name):
					return True
		return False


	def __str__(self):
		return ' '.join(map(str, self.variables()))


	def __repr__(self):
		mechs = ', '.join(f'{v.name}{"("+", ".join(v.parents)+")" if len(v.parents) else "()"}'
		                  for v in self.variables())
		return f'{self.__class__.__name__}({mechs})'




class BayesKit(DynamicKit, BayesLike, SCM):
	def __init__(self, mechanisms=None, **kwargs):
		super().__init__(**kwargs)
		if mechanisms is not None:
			self.include(*mechanisms)



class BayesNetContext(BayesLike, SCM, Context):
	def context(self, size=None, **kwargs):
		if size is None:
			size = self.size
		return BayesNetContext(size=size, **kwargs).include(*self.sources())


	@property
	def size(self):
		if self._size is None:
			return 10
		return self._size
	@size.setter
	def size(self, val):
		self._size = val



class BayesNet(BayesLike, SCM, Seeded, Structured):
	equation_type = 'bernoulli'

	def __init__(self, *, params=None, **kwargs):
		super().__init__(**kwargs)
		for var in self.unordered_variables():
			var.param = params[var.name] if params is not None and var.name in params \
				else var.generate_parameters(gen=self._rng)


	def has_variable(self, name: Union[str, AbstractVariable]) -> bool:
		return isinstance(getattr(self, name, None), AbstractVariable)


	def get_variable(self, name: str, default=unspecified_argument):
		var = getattr(self, name, default)
		if var is unspecified_argument:
			raise AttributeError(name)
		return var


	def get_from(self, ctx: Optional['AbstractContext'], gizmo: str):
		return self.get_variable(gizmo).get_from(ctx, gizmo)


	@classmethod
	def get_static_variable(cls, name: str, default=unspecified_argument):
		return getattr(cls, name) if default is unspecified_argument else getattr(cls, name, default)


	@classmethod
	def static_unordered_variables(cls):
		for loc, name, tool in cls._emit_all_craft_items():
			if isinstance(tool, AbstractVariable):
				yield tool



class IndependentBernoullis(BayesKit):
	def __init__(self, **terms: float):
		super().__init__(BernoulliMechanism(name=label, param=param) for label, param in terms.items())



# see examples in test/test_bayes.py

