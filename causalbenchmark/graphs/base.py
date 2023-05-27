from typing import Type, Any, Dict, Union, Iterator, Optional, Callable, Tuple, List, Sequence, FrozenSet, Iterable, Set, Mapping
from omnibelt import Class_Registry, unspecified_argument, JSONABLE
import numpy as np
import networkx as nx
import pomegranate as pg
from functools import lru_cache
from omniplex import hparam
# from omnidata import Structured, material as _material, machine as _machine, hparam, inherit_hparams
# source = _material.from_size # just an alias to make the SCMs more legible
# mechanism = _machine # just an alias to make the SCMs more legible

from .. import util
from ..util import generate_all_bit_strings, Seeded



_graph_registry = Class_Registry()
_register_graph = _graph_registry.get_decorator()
get_graph_class = _graph_registry.get_class


class register_graph(_register_graph):
	def __init__(self, name, *args, **kwargs):
		super().__init__(name, *args, **kwargs)
		self._name = name


	def __call__(self, cls):
		if getattr(cls, '_graph_name', None) is None:
			setattr(cls, '_graph_name', self._name)
		return super().__call__(cls)



def create_graph(graph_name, *args, **kwargs):
	return get_graph_class(graph_name)(*args, **kwargs)



class AbstractVariable:
	@property
	def name(self) -> str:
		raise NotImplementedError


	@property
	def parents(self) -> Tuple[str]:
		raise NotImplementedError


	@property
	def param(self) -> Union[np.ndarray, int, float]:
		raise NotImplementedError
	@param.setter
	def param(self, val: Union[np.ndarray, int, float]):
		raise NotImplementedError


	def prob(self, val=1, **conds) -> float:
		raise NotImplementedError


	def parameter_index(self, **conds) -> int:
		raise NotImplementedError


	def dof(self) -> int:
		raise NotImplementedError



class AbstractProcess:
	def has_variable(self, name: str) -> bool:
		raise NotImplementedError


	def get_variable(self, name: str, default=unspecified_argument):
		raise NotImplementedError


	def unordered_variables(self) -> Iterator[AbstractVariable]:
		raise NotImplementedError


	@classmethod
	def static_unordered_variables(cls) -> Iterator[AbstractVariable]:
		raise NotImplementedError


	def variables(self) -> Iterator[AbstractVariable]:
		yield from self._topo_order(self.unordered_variables())


	def variable_names(self) -> Iterator[str]:
		for v in self.variables():
			yield v.name


	def source_variables(self) -> Iterator[AbstractVariable]:
		'''returns all the variables that have no parents'''
		for v in self.variables():
			if not len(v.parents):
				yield v


	@classmethod
	def get_static_variable(cls, name: str, default=unspecified_argument):
		raise NotImplementedError


	@classmethod
	def static_variables(cls) -> Iterator[AbstractVariable]:
		yield from cls._topo_order(cls.static_unordered_variables())


	@classmethod
	def static_variable_names(cls) -> Iterator[str]:
		for v in cls.static_variables():
			yield v.name


	@classmethod
	def static_source_variables(cls) -> Iterator[AbstractVariable]:
		'''returns all the variables that have no parents'''
		for v in cls.static_variables():
			if not len(v.parents):
				yield v


	@staticmethod
	def _topo_order(unordered_node_iter: Iterable[AbstractVariable]) -> Iterator[AbstractVariable]:
		'''lists all the same nodes as in unordered_variables, but in topological order'''
		waiting = []
		done = set()
		result = []

		for node in unordered_node_iter:
			if len(node.parents) == 0 or all(p in done for p in node.parents):
				result.append(node)
				done.add(node.name)

			else:
				waiting.append(node)

		while result:
			node = result.pop(0)
			yield node
			for w in waiting:
				if len(w.parents) == 0 or all(p in done for p in w.parents):
					result.append(w)
					done.add(w.name)
					waiting.remove(w)



class ExplicitProcess(AbstractProcess):
	@classmethod
	@lru_cache(maxsize=50)
	def dof(cls) -> int:
		return sum(v.dof() for v in cls.static_variables())


	@classmethod
	def from_parameters(cls, params): # TODO: avoid generating params before replacing them
		return cls().set_parameters(params)


	def set_parameters(self, params: np.ndarray):
		offset = 0
		for v in self.variables():
			N = v.dof()
			v.param = params[offset:offset+N]
			offset += N
		return self


	def get_parameters(self):
		return np.concatenate([v.param.flat for v in self.variables()])


	@classmethod
	@lru_cache(maxsize=50)
	def _parameter_index_offset(cls, name: str):
		offset = 0
		for node in cls.static_variables():
			if node.name == name:
				return offset
			offset += node.dof()


	@classmethod
	def parameter_index(cls, name: str, **parents: int) -> int:
		offset = cls._parameter_index_offset(name)
		index = offset + cls.get_static_variable(name).parameter_index(**parents)
		return index


class StatisticalProcess(AbstractProcess):
	def correlation(self, var1: str, var2: str, **conditions: int) -> float:
		raise NotImplementedError


	def probability(self, **evidence: int) -> float:
		raise NotImplementedError


	def distribution(self, variable: str, *parents: str, **conditions: int) -> AbstractVariable:
		raise NotImplementedError



class CausalProcess(StatisticalProcess):
	# region Full Bounds
	def full_ate_bounds(self, treatment: str, *, treated: bool = True) -> Dict[str, Tuple[float, float]]:
		raise NotImplementedError


	def full_ett_bounds(self, treatment: str, *, treated: bool = True) -> Dict[str, Tuple[float, float]]:
		raise NotImplementedError


	def full_nde_bounds(self, treatment: str, outcome: str, *, mediators: Optional[Sequence[str]] = None,
	                    treated: bool = True) -> Dict[str, Tuple[float, float]]:
		raise NotImplementedError


	def full_nie_bounds(self, treatment: str, outcome: str, *, mediators: Optional[Sequence[str]] = None,
	                    treated: bool = True) -> Dict[str, Tuple[float, float]]:
		raise NotImplementedError


	def full_marginal_bounds(self, **conditions: str) -> Dict[str, Tuple[float, float]]:
		raise NotImplementedError


	def full_interventional_bounds(self, interventions: Dict[str, float], *,
	                          conditions: Optional[Dict[str, int]] = None) -> Dict[str, Tuple[float, float]]:
		raise NotImplementedError


	def full_counterfactual_bounds(self, *, factual: Optional[Dict[str, int]] = None,
	                          evidence: Optional[Dict[str, int]] = None,
	                          action: Optional[Dict[str, float]] = None) -> Dict[str, Tuple[float, float]]:
		raise NotImplementedError
	# endregion


	# region Bounds
	def ate_bounds(self, outcome: str, treatment: str, *, treated: bool = True) -> Tuple[float, float]:
		return self.full_ate_bounds(treatment, treated=treated)[outcome]


	def ett_bounds(self, outcome: str, treatment: str, *, treated: bool = True) -> Tuple[float, float]:
		return self.full_ett_bounds(treatment, treated=treated)[outcome]


	def nde_bounds(self, outcome: str, treatment: str, *, mediators: Optional[Sequence[str]] = None,
	               treated: bool = True) -> Tuple[float, float]:
		return self.full_nde_bounds(treatment, outcome, mediators=mediators, treated=treated)[outcome]


	def nie_bounds(self, outcome: str, treatment: str, *, mediators: Optional[Sequence[str]] = None,
	               treated: bool = True) -> Tuple[float, float]:
		return self.full_nie_bounds(treatment, outcome, mediators=mediators, treated=treated)[outcome]


	def marginal_bounds(self, outcome: str, **conditions: str) -> Tuple[float, float]:
		return self.full_marginal_bounds(**conditions)[outcome]


	def interventional_bounds(self, outcome: str, interventions: Dict[str, float], *,
	                          conditions: Optional[Dict[str, int]] = None) -> Tuple[float, float]:
		return self.full_interventional_bounds(interventions, conditions=conditions)[outcome]


	def counterfactual_bounds(self, outcome: str, *, factual: Optional[Dict[str, int]] = None,
	                          evidence: Optional[Dict[str, int]] = None,
	                          action: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
		return self.full_counterfactual_bounds(factual=factual, evidence=evidence, action=action)[outcome]
	# endregion


	def intervene(self, **interventions: float) -> 'CausalProcess':
		raise NotImplementedError



# class SymbolicCausalProcess(CausalProcess):
	@lru_cache(maxsize=50)
	def symbolic_graph_structure(self) -> str:
		edges = []
		for v in self.variables():
			for p in v.parents:
				edges.append(f'{p}->{v.name}')
		return ','.join(edges)


	def symbolic_mechanisms(self) -> str:
		return str(self)


	def symbolic_mechanism(self, name: str) -> str:
		return str(self.get_variable(name))


	def symbolic_marginal(self, outcome: str, **conditions: str) -> str:
		raise NotImplementedError


	def symbolic_interventional(self, outcome: str, interventions: Dict[str, float], *,
	                             conditions: Optional[Dict[str, int]] = None) -> str:
		raise NotImplementedError


	def symbolic_counterfactual(self, outcome: str, *, factual: Optional[Dict[str, int]] = None,
	                             evidence: Optional[Dict[str, int]] = None,
	                             action: Optional[Dict[str, float]] = None) -> str:
		raise NotImplementedError


	def symbolic_ate_estimand(self, outcome: str, treatment: str, *, treated: bool = True) -> str:
		raise NotImplementedError


	def symbolic_ett_estimand(self, outcome: str, treatment: str, *, treated: bool = True) -> str:
		raise NotImplementedError


	def symbolic_nde_estimand(self, outcome: str, treatment: str, *, mediators: Optional[Sequence[str]] = None,
	                           treated: bool = True) -> str:
		raise NotImplementedError


	def symbolic_nie_estimand(self, outcome: str, treatment: str, *, mediators: Optional[Sequence[str]] = None,
	                           treated: bool = True) -> str:
		raise NotImplementedError


	def symbolic_ate_given_info(self, outcome: str, treatment: str, *, treated: bool = True) -> Dict[str, JSONABLE]:
		raise NotImplementedError


	def symbolic_ett_given_info(self, outcome: str, treatment: str, *, treated: bool = True) -> Dict[str, JSONABLE]:
		raise NotImplementedError


	def symbolic_nde_given_info(self, outcome: str, treatment: str, *, mediators: Optional[Sequence[str]] = None,
	                            treated: bool = True) -> Dict[str, JSONABLE]:
		raise NotImplementedError


	def symbolic_nie_given_info(self, outcome: str, treatment: str, *, mediators: Optional[Sequence[str]] = None,
	                            treated: bool = True) -> Dict[str, JSONABLE]:
		raise NotImplementedError


	def symbolic_marginal_given_info(self, outcome: str) -> Dict[str, JSONABLE]:
		raise NotImplementedError


	def symbolic_ate_variables(self, outcome: str, treatment: str) -> List[str]:
		raise NotImplementedError


	def symbolic_ett_variables(self, outcome: str, treatment: str) -> List[str]:
		raise NotImplementedError


	def symbolic_nde_variables(self, outcome: str, treatment: str, *,
	                           mediators: Optional[Sequence[str]] = None) -> List[str]:
		raise NotImplementedError


	def symbolic_nie_variables(self, outcome: str, treatment: str, *,
	                           mediators: Optional[Sequence[str]] = None) -> List[str]:
		raise NotImplementedError


	def symbolic_marginal_variables(self, outcome: str) -> List[str]:
		raise NotImplementedError


	def backdoor_adjustment_sets(self, outcome: str, treatment: str) -> Iterator[FrozenSet[str]]:
		raise NotImplementedError
	
	
	def frontdoor_adjustment_sets(self, outcome: str, treatment: str) -> Iterator[FrozenSet[str]]:
		raise NotImplementedError
	
	
	def iv_adjustment_sets(self, outcome: str, treatment: str) -> Iterator[FrozenSet[str]]:
		raise NotImplementedError


# class Verbalization(CausalProcess):

	def variable_mapping(self, labels: Dict[str, str]):
		return dict(self._variable_mapping(labels))


	def _variable_mapping(self, labels: Dict[str, str]):
		for var in self.variables():
			label = labels.get(f'{var.name}name', None)
			yield f'{var.name}name', label if isinstance(label, str) else var.name
			label = labels.get(f'{var.name}1_noun', None)
			yield f'{var.name}1', label if isinstance(label, str) else f'{var.name}=1'
			label = labels.get(f'{var.name}0_noun', None)
			yield f'{var.name}0', label if isinstance(label, str) else f'{var.name}=0'


	def verbalize_background(self, labels: Dict[str, str]) -> str:
		raise NotImplementedError


	def story_variable_mapping(self, labels: Dict[str, str]):
		'''Returns a mapping from variable names to their labels'''
		return {key: val for var in self.variables()
		        for key, val in {f'{var.name}name': labels[f'{var.name}name'],
		                         f'{var.name}1': labels[f'{var.name}1_noun'],
								 f'{var.name}0': labels[f'{var.name}0_noun']}.items()}


	def verbalize_description(self, labels: Dict[str, str],
	                          mechanisms: Optional[Sequence[Union[str, AbstractVariable]]] = None) -> str:
		'''
		Describe the mechanisms of the graph using the given labels

		By default, this will describe all mechanisms. If you want to describe only a subset of mechanisms,
		pass in the names of the variables you want to describe.

		'''
		if mechanisms is None:
			mechanisms = list(self.variables())

		return ' '.join(self.verbalize_mechanism(mechanism, labels) for mechanism in mechanisms)


	def verbalize_mechanism(self, var: Union[str, AbstractVariable], labels: Dict[str, str]) -> str:
		'''Describe the mechanism of the given variable using the given labels'''
		raise NotImplementedError


	def verbalize_ate_given_info(self, labels: Dict[str, str], outcome: str, treatment: str) -> str:
		return self.verbalize_description(labels, mechanisms=self.symbolic_ate_variables(outcome, treatment))


	def verbalize_ett_given_info(self, labels: Dict[str, str], outcome: str, treatment: str) -> str:
		return self.verbalize_description(labels, mechanisms=self.symbolic_ett_variables(outcome, treatment))


	def verbalize_nde_given_info(self, labels: Dict[str, str], outcome: str, treatment: str) -> str:
		return self.verbalize_description(labels, mechanisms=self.symbolic_nde_variables(outcome, treatment))


	def verbalize_nie_given_info(self, labels: Dict[str, str], outcome: str, treatment: str) -> str:
		return self.verbalize_description(labels, mechanisms=self.symbolic_nie_variables(outcome, treatment))


	def verbalize_marginal_given_info(self, labels: Dict[str, str], outcome: str) -> str:
		return self.verbalize_description(labels, mechanisms=self.symbolic_marginal_variables(outcome))



class SCM(CausalProcess, ExplicitProcess):
# class SCM(SymbolicCausalProcess, ExplicitProcess, Verbalization):
	_graph_name = None
	_equation_type = None


	def to_networkx(self):
		G = nx.DiGraph()
		G.add_nodes_from(self.variable_names())
		for v in self.variables():
			for p in v.parents:
				G.add_edge(p, v.name)
		return G


	def structural_equation_parameters(self) -> Dict[str, JSONABLE]:
		return {v.name: v.param.tolist() for v in self.variables()}


	def meta_data(self) -> Dict[str, JSONABLE]:
		if util.LEGACY_MODE:
			return {
				'phenomenon': self._graph_name,
				'structuralEqs_type': self._equation_type,
				'structuralEqs': self.structural_equation_parameters(),

				'nodes': [v.name for v in self.variables()],

				'graph': self.symbolic_graph_structure(),

				'simpson': self._graph_name == 'simpson',

			}

		return {
			'phenomenon': self._graph_name,
			'structuralEqs_type': self._equation_type,
			'structuralEqs': self.structural_equation_parameters(),

			'nodes': [v.name for v in self.variables()],

			'edges': self.symbolic_graph_structure(),

		}


	def mechanism_parameters(self) -> Dict[str, JSONABLE]:
		raise NotImplementedError


	background_prefix = hparam('Imagine a self-contained, hypothetical world with only the following conditions, '
	                           'and without any unmentioned factors or causal relationships:', inherit=True)
	background_edge_template = hparam('{parent} has a direct effect on {children}.', inherit=True)
	background_unobserved_template = hparam('{vname} is unobserved.', inherit=True)
	unobserved_variables = hparam(None, inherit=True)


	def verbalize_graph(self, labels: Dict[str, str]) -> str:
		'''Describes the graph structure (nodes and edges) using the given labels'''
		lines = []
		edges = {}
		for v in self.variables():
			for p in v.parents:
				edges.setdefault(p, []).append(v.name)

		for parent, children in edges.items():
			lines.append(util.pformat(self.background_edge_template, parent=labels[f'{parent}name'], #children=children,
			                          # verbalized_list=self._verbalized_terms,
			                          children=util.verbalize_list([labels[f'{c}name'] for c in children]),
			                          **labels))

		return ' '.join(line[0].upper() + line[1:] for line in lines if len(line) > 0)


	def verbalize_background(self, labels: Dict[str, str]) -> str:
		lines = []
		if len(self.background_prefix) > 0:
			lines.append(self.background_prefix)

		lines.append(self.verbalize_graph(labels))

		if self.unobserved_variables is not None:
			for v in self.unobserved_variables:
				lines.append(self.background_unobserved_template.format(vname=labels[f'{v}name']))

		return ' '.join(line[0].upper() + line[1:] for line in lines if len(line) > 0)


	# region Mechanism Templates
	mechanism0_template = hparam('The overall probability of {{v.name}1_noun} is {v.param.item():.0%}.', inherit=True)
	mechanism1_template = hparam('For {{v.parents[0]}0_wheresentence}, the probability of {{v.name}1_noun} is {v.param[0]:.0%}. '
	                             'For {{v.parents[0]}1_wheresentence}, the probability of {{v.name}1_noun} is {v.param[1]:.0%}.',
	                             inherit=True)
	mechanism2_template = hparam('For {{v.parents[0]}0_wheresentence} and {{v.parents[1]}0_wherepartial}, '
	                                'the probability of {{v.name}1_noun} is {v.param[0,0]:.0%}. '
	                             'For {{v.parents[0]}0_wheresentence} and {{v.parents[1]}1_wherepartial}, '
	                                'the probability of {{v.name}1_noun} is {v.param[0,1]:.0%}. '
	                             'For {{v.parents[0]}1_wheresentence} and {{v.parents[1]}0_wherepartial}, '
	                                'the probability of {{v.name}1_noun} is {v.param[1,0]:.0%}. '
	                             'For {{v.parents[0]}1_wheresentence} and {{v.parents[1]}1_wherepartial}, '
	                                'the probability of {{v.name}1_noun} is {v.param[1,1]:.0%}.',
	                             inherit=True)
	mechanism3_template = hparam('For {{v.parents[0]}0_wheresentence}, {{v.parents[1]}0_wherepartial}, and {{v.parents[2]}0_wherepartial}, '
	                                'the probability of {{v.name}1_noun} is {v.param[0,0,0]:.0%}. '
	                             'For {{v.parents[0]}0_wheresentence}, {{v.parents[1]}0_wherepartial}, and {{v.parents[2]}1_wherepartial}, '
	                                'the probability of {{v.name}1_noun} is {v.param[0,0,1]:.0%}. '
	                             'For {{v.parents[0]}0_wheresentence}, {{v.parents[1]}1_wherepartial}, and {{v.parents[2]}0_wherepartial}, '
	                                'the probability of {{v.name}1_noun} is {v.param[0,1,0]:.0%}. '
	                             'For {{v.parents[0]}0_wheresentence}, {{v.parents[1]}1_wherepartial}, and {{v.parents[2]}1_wherepartial}, '
	                                'the probability of {{v.name}1_noun} is {v.param[0,1,1]:.0%}. '
	                             'For {{v.parents[0]}1_wheresentence}, {{v.parents[1]}0_wherepartial}, and {{v.parents[2]}0_wherepartial}, '
	                                'the probability of {{v.name}1_noun} is {v.param[1,0,0]:.0%}. '
	                             'For {{v.parents[0]}1_wheresentence}, {{v.parents[1]}0_wherepartial}, and {{v.parents[2]}1_wherepartial}, '
	                                'the probability of {{v.name}1_noun} is {v.param[1,0,1]:.0%}. '
	                             'For {{v.parents[0]}1_wheresentence}, {{v.parents[1]}1_wherepartial}, and {{v.parents[2]}0_wherepartial}, '
	                                'the probability of {{v.name}1_noun} is {v.param[1,1,0]:.0%}. '
	                             'For {{v.parents[0]}1_wheresentence}, {{v.parents[1]}1_wherepartial}, and {{v.parents[2]}1_wherepartial}, '
	                                'the probability of {{v.name}1_noun} is {v.param[1,1,1]:.0%}.',
	                             inherit=True)
	# endregion

	def verbalize_mechanism(self, var: Union[str, AbstractVariable], labels: Dict[str, str]) -> str:
		'''Describe the mechanism of the given variable using the given labels'''

		v = self.get_variable(var) if isinstance(var, str) else var

		templates = {0: self.mechanism0_template,
		             1: self.mechanism1_template,
		             2: self.mechanism2_template,
		             3: self.mechanism3_template
		             }

		if len(v.parents) not in templates:
			raise NotImplementedError(f'Cannot verbalize mechanisms with {len(v.parents)} parents: {v}')

		return util.pformat(templates[len(v.parents)], v=v, **labels)



















# class Graph(Structured): # TODO: future
# 	description_template = hparam(None)
#
# 	'''Automatically registers subclasses with the scm_registry (if a name is provided).'''
# 	def __init_subclass__(cls, name=None, **kwargs):
# 		super().__init_subclass__(**kwargs)
# 		if name is not None:
# 			register_graph(name)(cls)
# 		cls._graph_name = name
#



########################################################################################################################


# class AbductionFailed(ValueError):
# 	'''
# 	Raised if any source variables (parents) are not cached in the context
# 	(and consequently can't be extracted for counterfactuals).
# 	'''
# 	pass


# class SCM(Structured):
# 	description_template = hparam(None)
#
# 	'''Automatically registers subclasses with the scm_registry (if a name is provided).'''
# 	def __init_subclass__(cls, name=None, **kwargs):
# 		super().__init_subclass__(**kwargs)
# 		if name is not None:
# 			register_scm(name)(cls)
#
#
# 	def background(self, labels):
# 		return {'description': self.description(labels)}
#
#
# 	def description(self, labels):
# 		'''
# 		Generates the background context that precedes each question.
# 		Usually this context contains a general description of how the SCM works.
# 		'''
# 		if self.description_template is None:
# 			raise NotImplementedError
# 		return self.description_template.format(**labels)
#
#
# 	def all_variables(self): # just an alias
# 		yield from self.gizmos()
	
	
	# def source_variables(self):
	# 	for gizmo in self.gizmos():
	# 		for vendor in self.vendors(gizmo):
	# 			if _material.is_derivative(vendor):
	# 				yield gizmo
	
	
	# _AbductionFailed = AbductionFailed
	# def abduct_samples(self, ctx, *, require_cached=True):
	# 	sources = {}
	# 	for gizmo in self.source_variables():
	# 		if not ctx.is_cached(gizmo) and require_cached:
	# 			raise self._AbductionFailed(f'Gizmo {gizmo} is not cached in {ctx}')
	# 		sources[gizmo] = ctx[gizmo]
	# 	return sources



# class DeterministicSCM(SCM):
# 	def brute_force(self):
# 		srcs = list(self.source_variables())
# 		if len(srcs) > 10:
# 			raise ValueError('Too many sources to brute force')
# 		bit_strings = generate_all_bit_strings(len(srcs))
# 		return dict(zip(srcs, bit_strings.T))

