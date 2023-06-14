from typing import Sequence, Optional, List, Union, Iterator, Tuple
from scipy.spatial.distance import squareform
from scipy import optimize as opt
from functools import lru_cache
import numpy as np
from scipy import stats, special
import itertools

from .base import Seeded
from .. import util



class JointDistribution(Seeded):
	def __init__(self, N=None, *, params=None, **kwargs):
		assert N is not None or params is not None, 'Either N or params must be specified'
		if params is not None:
			N = len(params.shape)
		super().__init__(**kwargs)
		self._params = self._generate_params(N, params=params)


	@property
	def N(self):
		return len(self._params.shape)


	def _generate_params(self, N, params=None):
		if params is None:
			raw = self._rng.uniform(size=[2**N+1]).cumsum()
			raw /= raw[-1]
			params = raw[1:] - raw[:-1]
		else:
			params = np.asarray(params)
			assert params.size == 2**N
		assert np.isclose(params.sum(), 1), f'params must sum to 1, got {params.sum()}'
		return params.reshape([2]*N)


	def prob(self, *vals: Optional[float]) -> float:
		params = self._params
		assert len(vals) == len(params.shape), f'Expected {len(params.shape)} values, got {len(vals)}'
		N = len(vals)

		for i, val in enumerate(vals):
			if val is None:
				continue
			x = np.array([1 - val, val]).reshape([1] * i + [2] + [1] * (N - i - 1))
			params = params * x
			params = params.sum(axis=i, keepdims=True)

		return params.sum()


	def marginals(self, *conds: Optional[float], val: int = 1):
		if not len(conds):
			return np.asarray([self.marginal_index(i, val=val) for i in range(self.N)])
		assert len(conds) == self.N, f'Expected {self.N} values, got {len(conds)}'
		condition = self.prob(*conds)
		return np.asarray([self.marginal_index(i, val=val) / condition if c is None else c for i, c in enumerate(conds)])


	def marginal(self, *vals):
		sel = [slice(None) if val is None else val for val in vals]
		return self._params[tuple(sel)].sum()


	def marginal_index(self, *indices: int, val: int = 1) -> float:
		vals = [None] * self.N
		for i in indices:
			vals[i] = val
		return self.marginal(*vals)


	def variance(self, i: int) -> float:
		marginal = self.marginal_index(i)
		return marginal * (1 - marginal)


	def covariance(self, i: int, j: int) -> float:
		marginal_i = self.marginal_index(i)
		marginal_j = self.marginal_index(j)
		joint = self.marginal_index(i, j)
		return joint - marginal_i * marginal_j


	def correlation(self, i: int, j: int) -> float:
		covariance = self.covariance(i, j)
		variance_i = self.variance(i)
		variance_j = self.variance(j)
		return covariance / np.sqrt(variance_i * variance_j)


	def cov(self):
		return np.asarray([self.covariance(i, j) for i in range(self.N) for j in range(i + 1, self.N)])


	def corr(self):
		return np.asarray([self.correlation(i, j) for i in range(self.N) for j in range(i + 1, self.N)])


	def cov_matrix(self):
		return squareform(self.cov())


	def corr_matrix(self):
		return squareform(self.corr())



def power_set(options: Union[Sequence, int], *, min_size=None, max_size=None) -> Iterator[Tuple]:
	'''
	Generates all subsets of options, from min_size to max_size (inclusive).
	Starts from the smallest subsets (min_size) and works up to the largest (max_size).
	Including empty set and full set.

	:param options: set of objects to choose from
	:param min_size: of subsets (default 0)
	:param max_size: of subsets (default len(options))
	:return: generator of subsets
	'''
	if isinstance(options, int):
		options = list(range(options))
	if min_size is None:
		yield ()
	min_size = min_size or 1
	max_size = max_size or len(options)
	for r in range(min_size, max_size + 1):
		combinations = itertools.combinations(options, r)
		yield from combinations



def all_splits(options):
	'''
	Generates all possible splits of options into two groups.

	:param options: starting set
	:return: generator of (subset, complement) pairs such that subset + complement = options
	'''
	full = set(options)
	for sub in power_set(options):
		yield sub, tuple(full.difference(sub))



class Bernoulli_N_Body_Correlations:
	def __init__(self, N, *, max_order=None):
		self.N = N
		self.max_order = max_order or N

		codes = util.generate_all_bit_strings(N, dtype=bool).T  # shape (n, 2**n)

		def get_selection(inds):
			return codes[list(inds)].prod(axis=0).astype(bool)

		self.selections = {g: get_selection(g) for g in power_set(N, max_size=max_order) if len(g)}


	@staticmethod
	def max_entropy_setting(N: int) -> np.ndarray:
		'''
		Generates an array of all possible correlations for an N-dimensional Bernoulli distribution
		Starting with the first order correlations (equivalent to marginals), then pairwise correlations, etc.

		In the maximum entropy distribution, all marginals are 0.5 and higher order correlations are zero.

		:param N: number of dimensions
		:return: array of correlations shape (2**N - 1,)
		'''
		moments = np.zeros((2 ** N - 1,))
		moments[:N] = 0.5
		return moments


	def compute(self, probs) -> List[float]:
		'''
		Computes all n-body correlations for a multivariate Bernoulli distribution from the probabilities of each outcome.

		:param probs: probability of each possible outcome (2**N,) (assumed to sum to 1)
		:param max_order: maximum order of moments to compute (generally recommended to be 2)
		:return: list of moments
		'''
		N = self.N

		# intersections = {g: probs[get_selection(g)].sum() for g in power_set(N, max_size=max_order) if len(g)}
		intersections = {g: probs[sel].sum() for g, sel in self.selections.items()}

		marginals = np.asarray([intersections[(i,)] for i in range(N)])

		n_body_correlations = marginals.tolist()

		stds = np.sqrt(marginals * (1 - marginals))

		for group in power_set(N, min_size=2, max_size=self.max_order):
			terms = [
				(-1) ** len(marginal_indices)
				* intersections.get(moment_indices, 1.) # higher order moment
				* marginals[list(marginal_indices)].prod()  # if len(marginal_indices) else 1. # first order moments
				for marginal_indices, moment_indices in all_splits(group)]

			n_body_correlations.append(sum(terms) / stds[list(group)].prod())

		return n_body_correlations



def generate_joint_bernoulli(marginals, correlations, *, x0=None, max_order=None):
	N = len(marginals)

	moments = Bernoulli_N_Body_Correlations.max_entropy_setting(N)
	moments[:N] = marginals
	moments[N: N +len(correlations)] = correlations

	wts = np.ones((2 ** N - 1,)) * 1.
	wts[len(marginals) : len(marginals) + len(correlations)] = 5.
	wts[len(marginals ) +len(correlations):] = 1.

	return optimize_joint_bernoulli(moments, x0=x0, wts=wts, max_order=max_order)



def optimize_joint_bernoulli(moments, *, x0=None, wts=None, max_order=None, p=2.):
	# moments : (2**N-1,)
	N = int(np.round(np.log2(len(moments ) +1)))

	marginals = moments[:N]
	assert np.all(marginals >= 0) and np.all(marginals <= 1), f'invalid first moments: {marginals}'
	assert np.all(np.abs(moments[N:]) <= 1), f'invalid higher order moments: {moments[N:]}'

	if x0 is None:
		x0 = np.zeros((2 ** N,))

	constraints = [int(special.comb(N, r)) for r in range(1, N+ 1)]
	if max_order is not None:
		constraints = constraints[:max_order]
	num_constraints = sum(constraints)

	if wts is None:
		wts = np.ones((2 ** N - 1,))
	wts = np.asarray(wts)
	wts = wts[:num_constraints]

	targets = moments[:num_constraints].copy()

	system = Bernoulli_N_Body_Correlations(N, max_order=max_order)

	def step(params):
		probs = special.softmax(params)  # .reshape([2] * N)
		estimate = system.compute(probs)
		estimate = np.asarray(estimate)

		error = np.abs(estimate - targets) ** p
		loss = wts @ error
		return loss

	sol = opt.minimize(step, x0)

	return special.softmax(sol.x).reshape(([2] * N))



def test_multivariate_bernoulli():
	gen = np.random.RandomState(11)

	N = 3

	max_order = None
	# max_order = 2

	####################

	marginals = gen.uniform(0.1, 0.9, size=N)
	# marginals = np.ones(N) * 0.5

	correlations = 2 * gen.uniform(size=N * (N - 1) // 2) - 1
	# correlations = np.ones(N * (N - 1) // 2) * -0.9
	# correlations[0] = 0.2
	correlations = np.abs(correlations)

	params = generate_joint_bernoulli(marginals, correlations, max_order=max_order)

	####################

	moments = Bernoulli_N_Body_Correlations.max_entropy_setting(N)
	moments[:N] = marginals
	moments[N:N + len(correlations)] = correlations
	# params = optimize_joint_bernoulli(moments, max_order=max_order)

	####################

	D = JointDistribution(params=params)

	mar = marginals
	amr = D.marginals()

	cor = correlations
	acr = D.corr()

	targets = \
		moments.tolist()
	final = \
		Bernoulli_N_Body_Correlations(N, max_order=max_order).compute(params.reshape(-1))

	# samples = gen.choice(np.arange(2 ** N).astype(int), p=params.reshape(-1), size=50000)

	assert np.allclose(mar, amr, atol=1e-2), f'marginals: {mar} != {amr}'
	assert np.allclose(cor, acr, atol=1e-2), f'correlations: {cor} != {acr}'




















