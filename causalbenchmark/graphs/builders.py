from typing import Union, Dict, List, Iterator, Tuple, Optional, Any, Iterable, Callable, Sequence, Set, TypeVar, Type, cast
import numpy as np
from itertools import product
from functools import lru_cache
from pathlib import Path

import omnifig as fig

from omnibelt import unspecified_argument, JSONABLE
from omniply import Parameterized, hparam, submodule

from scipy.linalg import pascal
from scipy import optimize as opt

from .. import util
from ..util import Seeded, vocabulary

from .base import get_graph_class, create_graph
from .bayes import BayesLike, BayesNet
from .ensembles import SCMEnsemble
from .constrained import ConstrainedSCM



class AbstractBuilder:
	name = None

	@property
	def is_deterministic(self):
		'''asks whether the generated specs fully determines the SCM'''
		return True


	@classmethod
	def _find_graph_type(cls, story: Union[Dict[str, Any], str]) -> Type[BayesNet]:
		if isinstance(story, dict):
			return cls._find_graph_type(story['phenomenon'])
		return get_graph_class(story)


	def spec_count(self, story: Dict[str, Any]):
		raise NotImplementedError


	def sample_specs(self, story: Dict[str, Any],  N: int, *, allow_repeats=False):  # uniformly sample N specs
		max_N = self.spec_count(story)
		if not allow_repeats and self.is_deterministic and not max_N is None and N > max_N:
			raise ValueError(f'Cannot sample {N} specs from {max_N} available')

		while max_N is not None and N > max_N:
			yield from self.spawn_specs(story)
			N -= max_N
		yield from self._sample_specs_without_replacement(story, N)


	def _sample_specs_without_replacement(self, story: Dict[str, Any], N: int):
		raise NotImplementedError


	def spawn_specs(self, story: Dict[str, Any]) -> Iterator[JSONABLE]:
		raise NotImplementedError


	def generate_scm(self, story: Dict[str, Any], spec: JSONABLE, **scm_kwargs):
		raise NotImplementedError


	def verbalize_spec(self, labels: Dict[str, str], story: str, spec: JSONABLE):
		raise NotImplementedError


	def meta_data(self, scm, story: Dict[str, Any], spec: JSONABLE):
		return {}



class InfiniteBuilder(AbstractBuilder):
	def spec_count(self, story: Dict[str, Any]):
		return None



class ConstraintBuilder(AbstractBuilder):
	epsilon = hparam(0., inherit=True)

	_default_constrained_type = ConstrainedSCM
	@hparam(inherit=True)
	def constrained_type(self):
		return self._default_constrained_type


	def generate_scm(self, story: Dict[str, Any], spec: JSONABLE, **scm_kwargs):
		return self.generate_constrained_scm(story, spec, **scm_kwargs)


	def generate_constrained_scm(self, story: Dict[str, Any], spec: JSONABLE, **scm_kwargs):
		graph_id = story if isinstance(story, str) else story['phenomenon']
		if not isinstance(graph_id, str):
			graph_id = graph_id[0]
		return self.constrained_type(graph_id=graph_id, spec=spec, builder=self,
		                             constraints=self.generate_scm_constraints(story, spec), **scm_kwargs)


	def generate_scm_constraints(self, story: Dict[str, Any], spec: JSONABLE):
		raise NotImplementedError



class ParameterBuilder(AbstractBuilder, Seeded):
	def generate_scm(self, story: Dict[str, Any], spec: JSONABLE, **scm_kwargs):
		return self.generate_scm_example(story, spec, **scm_kwargs)


	def generate_scm_example(self, story: Dict[str, Any], spec, **scm_kwargs):
		assert 'params' not in scm_kwargs, 'cannot specify params when using a spec'
		graph_type = self._find_graph_type(story)
		return graph_type(params=self.generate_scm_params(story, spec), **scm_kwargs)


	def generate_scm_params(self, story: Dict[str, Any], spec: JSONABLE):
		raise NotImplementedError



@fig.component('random')
class RandomBuilder(ParameterBuilder, InfiniteBuilder):
	name = 'random'
	'''A builder that generates SCMs with random parameters.'''
	@property
	def is_deterministic(self):
		return False


	def spawn_specs(self, story: Dict[str, Any]) -> Iterator[JSONABLE]:
		base = self._find_graph_type(story)
		while True:
			yield {node.name: self._rng.uniform(0, 1, node.dof()).tolist() for node in base.static_variables()}


	def _sample_specs_without_replacement(self, story: Dict[str, Any], N: int):
		gen = self.spawn_specs(story)
		for _ in range(N):
			yield next(gen)


	def generate_scm_params(self, story: Dict[str, Any], spec: JSONABLE):
		return spec



@fig.component('deterministic')
class DeterministicMechanismBuilder(Seeded, AbstractBuilder):
	name = 'deterministic'
	'''A builder that generates SCMs with deterministic mechanisms.'''

	conjunction = hparam(False)  # choose whether Y is a conjunction or disjunction of X and Z
	negation = hparam(False) # choose whether Y is a conjunction or disjunction of X and Z


	@classmethod
	def _find_graph_type(cls, story: Union[Dict[str, Any], str]) -> Type[BayesNet]:
		if isinstance(story, str) and not story.startswith('det-'):
			story = f'det-{story}'
		return super()._find_graph_type(story)


	@property
	def is_deterministic(self):
		return True


	def _sample_specs_without_replacement(self, story: Dict[str, Any], N: int):
		total = self.spec_count(story)
		assert total < 1000, 'Cannot sample without replacement from a large number of deterministic specs'

		gen = list(self.spawn_specs(story))

		return self._rng.choice(gen, N, replace=False)


	def spec_count(self, story: Dict[str, Any]):
		base = self._find_graph_type(story)
		return base.spec_count()


	def spawn_specs(self, story: Dict[str, Any]) -> Iterator[JSONABLE]:
		base = self._find_graph_type(story)
		yield from base.spawn_choices()


	def generate_scm(self, story: Dict[str, Any], spec: JSONABLE, **scm_kwargs):
		assert 'params' not in scm_kwargs, 'cannot specify params when using a spec'
		graph_type = self._find_graph_type(story)
		return graph_type(**spec, **scm_kwargs)




@fig.component('nice')
class NiceSCMBuilder(ParameterBuilder, Parameterized): # TODO
	'''A builder that generates SCMs with a nice parameters (selected from a small set of natural values).'''
	# prob_options = hparam([0.1, 0.25, 0.5, 0.75, 0.9])
	name = 'nice'
	prob_options = hparam([0.2, 0.5, 0.8])


	def spec_count(self, story: Dict[str, Any]):
		return len(self.prob_options) ** self._find_graph_type(story).dof()


	def spawn_specs(self, story: Dict[str, Any]) -> Iterator[JSONABLE]:
		base = self._find_graph_type(story)

		for inds in product(range(len(self.prob_options)), repeat=base.dof()):
			ind_iter = iter(inds)
			yield {node.name: [self.prob_options[next(ind_iter)] for _ in range(node.dof())]
			       for node in base.static_variables()}


	def _sample_specs_without_replacement(self, story: Dict[str, Any], N: int): # uniformly sample N specs
		total = self.spec_count(story)
		assert N <= total, f'cannot sample {N} specs from {total} options'
		past = set()

		base = self._find_graph_type(story)

		for _ in range(N):
			inds = None
			while inds is None or str(inds) in past:
				inds = self._rng.choice(len(self.prob_options), size=base.dof())
			past.add(str(inds))
			ind_iter = iter(inds)
			yield {node.name: [self.prob_options[next(ind_iter)] for _ in range(node.dof())]
			       for node in base.static_variables()}


	def generate_scm_params(self, graph_id: str, spec: JSONABLE):
		return spec



class EnsembleBuilder(ParameterBuilder):
	_default_ensemble_type = SCMEnsemble
	@hparam(inherit=True)
	def ensemble_type(self):
		return self._default_ensemble_type


	def generate_scm(self, story: Dict[str, Any], spec: JSONABLE, **scm_kwargs):
		return self.generate_ensemble_scm(story, spec, **scm_kwargs)


	def generate_ensemble_scm(self, story: Dict[str, Any], spec: JSONABLE, **scm_kwargs):
		return self.ensemble_type(story=story, spec=spec, builder=self, **scm_kwargs)



class _MechanismCorrelationBuilder(ParameterBuilder):

	limit = 0.01
	_default_gap = 0.2

	_prior_warning = False

	def _generate_source_params(self, options: Sequence[Tuple[float, float]]):
		if isinstance(options, int):
			if not self._prior_warning:
				# print('WARNING: passing an int to _generate_source_params is deprecated')
				self._prior_warning = True
			options = [(0.01, 0.2), (0.4, 0.6), (0.8, 0.99)][options]
		if not isinstance(options[0], (list, tuple)):
			options = [options]
		mn, mx = self._rng.choice(options) if len(options) > 1 else options[0]
		return self._rng.uniform(mn, mx)


	@staticmethod
	def _sample_gapped_numbers(gaps, bias=None):
		'''

		:param gaps:
		:param bias: how much to bias the sampling towards the edges
		(bias = len(gaps) should roughly be uniform - i think, and thats the default)
		:return:
		'''
		gaps = np.asarray(gaps)
		squeeze = False
		if gaps.ndim == 1:
			squeeze = True
			gaps = gaps.reshape(1, -1)
		gaps = gaps.T
		D, B = gaps.shape

		if bias is None:
			bias = D

		lims = np.zeros((D + 2, B))
		lims[1:-1] = gaps

		coverage = gaps.sum(0, keepdims=True)
		if np.any(coverage > 1):
			raise ValueError("Sum of gaps exceeds 1.")

		remaining = 1 - coverage

		spaces = np.random.uniform(0, 1, size=(D + 2, B))
		spaces[0] *= bias
		spaces[-1] *= bias

		spaces = spaces / spaces.sum(0, keepdims=True) * remaining

		numbers = spaces.cumsum(0) + lims.cumsum(0)

		samples = numbers[:-1].T
		if squeeze:
			samples = samples.squeeze()
		return samples


	@staticmethod
	def _find_blobs(arr):
		# Ensure that the array is a numpy array
		if type(arr) != np.ndarray:
			arr = np.array(arr)

		# Add zero at the start and at the end to handle edge blobs
		arr = np.concatenate(([0], arr, [0]))

		# Compute the differences between consecutive elements
		diffs = np.diff(arr)

		# Find the indices where blobs start (1) and end (-1)
		starts = np.where(diffs == 1)[0]
		ends = np.where(diffs == -1)[0]

		# Compute the sizes of the blobs
		sizes = ends - starts

		return starts.tolist(), sizes.tolist()


	def _generate_constrained_params(self, *constraints: float, bias=None):
		assert len(constraints) > 0, 'must have at least one constraint'
		assert sum(map(abs, constraints)) <= 1, f'constraints must sum to less than 1, got {constraints}'

		lims = np.zeros([2] * len(constraints))

		all = [slice(None)] * len(constraints)
		for i, diff in enumerate(constraints):
			sel = all.copy()
			sel[i] = 1
			lims[tuple(sel)] += diff
		lims -= lims.min()

		order = np.argsort(lims.reshape(-1))
		line = lims.reshape(-1)[order]
		gaps = line[1:] - line[:-1]
		samples = self._sample_gapped_numbers(gaps, bias=bias)
		gap_blobs = np.isclose(gaps, 0)
		gap_indices, gap_sizes = self._find_blobs(gap_blobs)
		indices = gap_indices
		sizes = [n+1 for n in gap_sizes]
		reorder = order.copy()
		for i, n in zip(indices, sizes):
			prev, new = np.arange(n) + i, np.random.permutation(n) + i
			reorder[new] = reorder[prev]
		inds = np.argsort(reorder)
		probs = samples[inds].reshape(lims.shape)
		return probs


	def _generate_mechanism_params(self, parents, spec, bias=None):
		constraints = [spec.get(p, 0.) for p in parents]
		constraints = [self._default_gap * c if abs(c)==1 else c for c in constraints]
		return self._generate_constrained_params(*constraints, bias=bias)


	def generate_scm_params(self, story: Dict[str, Any], spec: JSONABLE, limit=0.01, gap=0.2):
		nodes = list(self._find_graph_type(story).static_variables())

		params = {}

		for node in nodes:
			if node.name in spec:
				if len(node.parents) == 0:
					# if node.name not in spec:
					# 	# spec[node.name] = self._rng.integers(0, len(self.bin_labels))
					# 	raise ValueError(f'no spec for {node.name}')
					params[node.name] = self._generate_source_params(spec[node.name])
				else:
					params[node.name] = self._generate_mechanism_params(node.parents, spec.get(node.name, {}))

		return params



@fig.component('difficulty')
class DifficultyBuilder(InfiniteBuilder, _MechanismCorrelationBuilder):
	name = 'difficulty'
	difficulty = hparam('easy')

	@property
	def is_deterministic(self):
		return False


	def spec_count(self, story: Dict[str, Any]):
		assert self.difficulty in story, f'no difficulty {self.difficulty} in story {story}'
		return len(story[self.difficulty])


	def spawn_specs(self, story: Dict[str, Any]) -> Iterator[JSONABLE]:
		assert self.difficulty in story, f'no difficulty {self.difficulty} in story {story}'
		yield from story[self.difficulty]


	def _sample_specs_without_replacement(self, story: Dict[str, Any], N: int):
		if self.spec_count(story) > 1000:
			raise NotImplementedError(f'cannot sample without replacement from {self.spec_count(story)} specs')
		specs = list(self.spawn_specs(story))
		yield from self._rng.choice(specs, N, replace=False)


	def meta_data(self, scm, story: Dict[str, Any], spec: JSONABLE):
		return {
			'difficulty': self.difficulty,
			**super().meta_data(scm, story, spec),
		}

	# def generate_scm_params(self, story: Dict[str, Any], spec: JSONABLE):
	# 	limit = story.get('param-limit', 0.01)
	# 	return spec



@fig.component('mech-corr')
class MechanismCorrelationBuilder(ConstraintBuilder, EnsembleBuilder, _MechanismCorrelationBuilder, Parameterized):
	name = 'mech-corr'
	bin_labels = hparam(['much lower than', 'about the same as', 'much higher than'])
	@hparam
	def bins(self):
		return len(self.bin_labels)


	gap = hparam(0.20) # when variables are correlated, whats the gap between the probabilities
	limit = hparam(0.01) # the minimum probability of a parameter (and 1-limit is the max)


	force_correlation = hparam(True) # if true, each mechanism will always have at least 1 constraint
	use_causal_template = hparam(True) # if true, use the causal template for verbalizing the spec

	_anticausal_correlation_template = 'When {{var.name}1_sentence}, {{parent}{int(spec[var.name][parent] > 0)}_sentence} ' \
	                        'is always more likely than {{parent}{int(spec[var.name][parent] < 0)}_sentence}.'

	_causal_correlation_template = 'When {{parent}1_sentence}, {{var.name}{int(spec[var.name][parent] > 0)}_sentence} ' \
	                        'is always more likely than {{var.name}{int(spec[var.name][parent] < 0)}_sentence}.'

	_source_template = 'The chance of {{var.name}1_noun} is {bin_labels[spec[var.name]]} {{var.name}0_noun}.'
	def verbalize_spec(self, labels: Dict[str, str], story: str, spec: JSONABLE):
		lines = []

		for var in self._find_graph_type(story).static_variables():
			if len(var.parents) == 0:
				lines.append(util.pformat(self._source_template,
				                          var=var, spec=spec, bin_labels=self.bin_labels, **labels))
			else:
				for parent in var.parents:
					if parent not in spec or spec[var.name][parent] is None or spec[var.name][parent] == 0:
						continue

					lines.append(util.pformat(self._causal_correlation_template if self.use_causal_template
					                          else self._anticausal_correlation_template,
					                          parent=parent, var=var, spec=spec, bin_labels=self.bin_labels, **labels))

		return ' '.join(line[0].upper() + line[1:] for line in lines if len(line) > 0)


	@property
	def is_deterministic(self):
		return False


	# @staticmethod
	# @lru_cache(maxsize=None)
	# def bin_min_limits(bins, limit=0.01):
	# 	lims = np.linspace(limit, 1 - limit, bins * 2)
	# 	mn, mx = lims[::2], lims[1::2]
	# 	return mn.copy()
	#
	#
	# @staticmethod
	# @lru_cache(maxsize=None)
	# def bin_max_limits(bins, limit=0.01):
	# 	lims = np.linspace(limit, 1 - limit, bins * 2)
	# 	mn, mx = lims[::2], lims[1::2]
	# 	return mx.copy()
	#
	#
	# def _generate_source_params(self, sel):
	#
	# 	width = 1 - 2 * self.limit
	# 	assert width > 0, f'limit must be less than 0.5, got {self.limit}'
	#
	# 	# lims = np.linspace(self.limit, 1 - self.limit, self.bins * 2)
	# 	#
	# 	# mn, mx = lims[sel * 2], lims[sel * 2 + 1]
	# 	bias = self._rng.uniform(self.bin_min_limits(self.bins, self.limit)[sel],
	# 	                         self.bin_max_limits(self.bins, self.limit)[sel])
	# 	return bias


	def spec_count(self, story: Dict[str, Any]):
		choices = [3**len(node.parents) - int(self.force_correlation) if len(node.parents) else self.bins
		           for node in self._find_graph_type(story).static_variables()]
		return np.prod(choices).item()


	def _sample_specs_without_replacement(self, story: Dict[str, Any], N: int):
		if self.spec_count(story) > 1000:
			raise NotImplementedError(f'cannot sample without replacement from {self.spec_count(story)} specs')
		specs = list(self.spawn_specs(story))
		yield from self._rng.choice(specs, N, replace=False)


	# def sample_specs(self, story: Dict[str, Any], N: int, *, allow_repeats=False): # uniformly sample N specs
	# 	total = self.spec_count(story)
	# 	assert allow_repeats or N <= total, f'cannot sample {N} specs from {total} options'
	#
	# 	all = list(self.spawn_specs(story))
	# 	for _ in range(N):
	# 		ind = self._rng.choice(len(all))
	# 		yield all[ind]
	# 		if not allow_repeats:
	# 			del all[ind]


	def spawn_specs(self, story: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
		base = self._find_graph_type(story)

		# source nodes
		sources = [src.name for src in base.static_source_variables()]

		mechs = [node for node in base.static_variables() if len(node.parents) > 0]
		edges = sum(len(node.parents) for node in mechs)
		
		total = len(sources) + len(mechs)
		
		src_options = [self.bins] * len(sources)
		
		for sel in np.ndindex(*src_options):
			src_sel = sel[:len(sources)]
			src_spec = {src: sel for src, sel in zip(sources, src_sel)}

			
			for mech_sel in product([-self.gap, 0, self.gap], repeat=edges):
				mech_sel = iter(mech_sel)
				
				spec = src_spec.copy()
				for node in mechs:
					picks = {parent: next(mech_sel) for parent in node.parents}
					if self.force_correlation and not any(pick != 0 for pick in picks.values()):
						break
					else:
						spec[node.name] = picks
				
				if len(spec) == total:
					yield spec


	def _correlation_to_constraint(self, base, node, corrs: Dict[str, int]):
		Anew = []
		bnew = []

		for parent, corr in corrs.items():
			assert corr != 0, 'cannot have a correlation of 0'
			rest = [p for p in node.parents if p != parent]

			mag = abs(corr)
			if mag >= 1:
				mag = self.gap

			if len(rest) == 0:
				proj = np.zeros(base.dof())

				proj[base.parameter_index(node.name, **{parent: 1})] = 1
				proj[base.parameter_index(node.name, **{parent: 0})] = -1
				if corr < 0:
					proj *= -1

				Anew.append(proj)
				bnew.append(mag + self.epsilon)

			else:
				for conds in product([0, 1], repeat=len(rest)):
					conds = {p: c for p, c in zip(rest, conds)}

					proj = np.zeros(base.dof())

					proj[base.parameter_index(node.name, **{parent: 1}, **conds)] = 1
					proj[base.parameter_index(node.name, **{parent: 0}, **conds)] = -1
					if corr < 0:
						proj *= -1

					Anew.append(proj)
					bnew.append(mag + self.epsilon)

		return Anew, bnew


	def _source_to_constraint(self, base, node, bin: int):
		if isinstance(bin, int):
			mn, mx = [(0.01, 0.2), (0.4, 0.6), (0.8, 0.99)][bin]
		else:
			mn, mx = bin

		mn += self.epsilon
		mx -= self.epsilon

		N = base.dof()
		index = base.parameter_index(node.name)

		proj = np.zeros(N)
		proj[index] = 1
		return [proj, -proj], [mn, -mx]


	def generate_scm_constraints(self, story: Dict[str, Any], spec: JSONABLE):
		base = self._find_graph_type(story)
		n = base.dof()

		constraints = [opt.LinearConstraint(np.eye(n),
		                                    self.limit + self.epsilon,
		                                    1 - self.limit - self.epsilon,
		                                    keep_feasible=False)] # TODO: find method where this can be true

		b = []
		A = []

		for node in base.static_variables():
			if node.name in spec:
				if len(node.parents):
					corrs = {parent: val for parent, val in spec[node.name].items() if val != 0}
					if len(corrs):
						Anew, bnew = self._correlation_to_constraint(base, node, corrs)
						b.extend(bnew)
						A.extend(Anew)
				else:
					Anew, bnew = self._source_to_constraint(base, node, spec[node.name])
					b.extend(bnew)
					A.extend(Anew)

		if len(A):
			constraints.append(opt.LinearConstraint(np.stack(A), np.asarray(b), keep_feasible=False))
		return constraints



# @fig.modifier('anticommonsense')
# class AnticommonsenseModifier(AbstractBuilder):
# 	@hparam
# 	def path(self):
# 		return util.config_root() / 'new' / 'anticommonsense.yml'
# 	@path.formatter
# 	def path(self, value):
# 		return Path(value)
#
#
# 	def generate_scm_example(self, story: Dict[str, Any], spec, **scm_kwargs):
# 		assert 'params' not in scm_kwargs, 'cannot specify params when using a spec'
# 		graph_type = self._find_graph_type(story)
# 		return graph_type(params=self.generate_scm_params(story, spec), **scm_kwargs)
#
#
# 	def meta_data(self, scm, story: Dict[str, Any], spec: JSONABLE):
# 		return {'anticommonsense': True,
# 		        **super().meta_data(scm, story, spec)}


# class SpecBuilder(AbstractBuilder):
# 	graph_id = hparam(inherit=True)
# 	@hparam(required=True, inherit=True)
# 	def graph_type(self) -> BayesLike:
# 		return get_graph_class(self.graph_id)
#
# 	spec = hparam(required=True, inherit=True)
#
#
# 	def variables(self):
# 		yield from self.graph_type.static_variables()
#
#
# 	def generate_story_params(self, graph_type, spec):
# 		raise NotImplementedError
#
#
# 	def create(self, graph_type: Union[Type, str] = None, spec=None, **sample_kwargs):
# 		if graph_type is None:
# 			graph_type = self.graph_type
# 		if spec is None:
# 			spec = self.spec
# 		assert 'params' not in sample_kwargs, 'cannot specify params when using a spec'
# 		return super().create(graph_type, params=self.generate_story_params(graph_type, spec), **sample_kwargs)



# class RelativeBuilder(SpecBuilder):
# 	gap = hparam(0.25)
# 	limit = hparam(0.02)
#
# 	bins = hparam(3)
#
#
# 	@property
# 	def is_deterministic(self):
# 		return False
#
#
# 	def _generate_source_params(self, sel):
# 		width = 1 - 2 * self.limit
# 		assert width > 0, f'limit must be less than 0.5, got {self.limit}'
#
# 		lims = np.linspace(self.limit, 1-self.limit, self.bins * 2)
#
# 		mn, mx = lims[sel*2], lims[sel*2+1]
# 		bias = self._rng.uniform(mn, mx)
# 		return bias
#
#
# 	def _generate_constrained_params(self, parents: int, constraints: int = 0):
# 		if parents > 3:
# 			raise NotImplementedError('Only up to 3 parents are supported')
#
# 		width = 1 - 2 * self.limit
# 		assert width > 0, f'limit must be less than 0.5, got {self.limit}'
#
# 		if constraints == 0:
# 			return self._rng.uniform(self.limit, 1 - self.limit, 2 ** parents)
#
# 		assert constraints <= parents, f'cannot have more constraints than parents, ' \
# 		                               f'got {constraints} constraints for {parents} parents'
#
# 		gap = self.gap
# 		assert gap is None or gap*constraints < width, f'gap must be less than {width/constraints} ' \
# 		                                               f'(for {constraints} constraints), got {gap}'
# 		if gap is None:
# 			gap = 0.
#
# 		groups = pascal(constraints+1, kind='lower')[-1]
# 		groups = groups * 2 ** parents / groups.sum()
# 		groups = groups.astype(int)
#
# 		probs = []
#
# 		mx = 1 - self.limit - gap * constraints
# 		mn = self.limit
#
# 		for group in groups:
# 			new = self._rng.uniform(mn, mx, group)
# 			mn = new.max() + gap
# 			mx += gap
#
# 			probs.extend(new)
#
# 		return np.array(probs).reshape([2]*parents)
#
#
# 	def _generate_mechanism_params(self, parents, spec):
# 		constraints = sum(1 for c in spec.values() if c is not None and c != 0)
# 		raw = self._generate_constrained_params(len(parents), constraints)
#
# 		if constraints == 0: # not recommended: usually there are constraints
# 			return raw
#
# 		probs = raw
#
# 		order = []
# 		fine = 0
# 		constrained = 0
# 		flips = []
#
# 		for axis, parent in enumerate(parents):
# 			if parent not in spec or spec[parent] is None or spec[parent] == 0:
# 				order.append(constraints + fine)
# 				fine += 1
# 			else:
# 				if spec[parent] < 0:
# 					flips.append(axis)
# 				order.append(constrained)
# 				constrained += 1
#
# 		if len(order) > 1:
# 			probs = probs.transpose(order)
# 		probs = np.flip(probs, axis=flips)
# 		return probs
#
#
# 	def generate_story_params(self, graph_id, spec):
# 		nodes = list(create_graph(graph_id).nodes()) # TODO avoid instantiating a graph here
#
# 		params = {}
#
# 		for node in nodes:
# 			if len(node.parents) == 0:
# 				if node.name not in spec:
# 					# spec[node.name] = self._rng.integers(0, len(self.bin_labels))
# 					raise ValueError(f'no spec for {node.name}')
# 				params[node.name] = self._generate_source_params(spec[node.name])
# 			else:
# 				params[node.name] = self._generate_mechanism_params(node.parents, spec.get(node.name, {}))
#
# 		return params
#
#





