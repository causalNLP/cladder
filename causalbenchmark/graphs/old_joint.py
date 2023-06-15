from typing import Sequence, Optional, List, Union, Iterator, Tuple
from scipy.spatial.distance import squareform
from scipy import optimize as opt
from functools import lru_cache
import numpy as np
from scipy import stats, special

import itertools


from .base import Seeded
from .. import util


def prentice_bounds(marginals):
	N = len(marginals)
	I, J = np.triu_indices(N, k=1)
	pi, pj = marginals[I], marginals[J]
	lim = np.minimum(np.sqrt((pi * (1 - pj)) / (pj * (1 - pi))), np.sqrt((pj * (1 - pi)) / (pi * (1 - pj))))
	return lim



class BernoulliLoss:
	def __init__(self, N, **kwargs):
		super().__init__(**kwargs)
		self.N = N


	@property
	def n_constraints(self):
		raise NotImplementedError


	def marginal(self, params, *indices: int, val: int = 1) -> float:
		sel = [slice(None)] * len(params.shape)
		for i in indices:
			sel[i] = val
		return params[tuple(sel)].sum()


	def variance(self, params, i: int) -> float:
		marginal = self.marginal(params, i)
		return marginal * (1 - marginal)


	def covariance(self, params, i: int, j: int) -> float:
		marginal_i = self.marginal(params, i)
		marginal_j = self.marginal(params, j)
		joint = self.marginal(params, i, j)
		return joint - marginal_i * marginal_j


	def correlation(self, params, i: int, j: int) -> float:
		covariance = self.covariance(params, i, j)
		variance_i = self.variance(params, i)
		variance_j = self.variance(params, j)
		return covariance / np.sqrt(variance_i * variance_j)


	def compute_loss(self, params):
		raise NotImplementedError


	def estimate(self, params):
		raise NotImplementedError



class TargetLoss(BernoulliLoss):
	def __init__(self, targets, *, wts=None, p=2, N=None, **kwargs):
		if N is None:
			N = len(targets)
		if wts is None:
			wts = np.ones((N,))
		if not isinstance(wts, np.ndarray):
			wts = np.asarray(wts)
		if not isinstance(targets, np.ndarray):
			targets = np.asarray(targets)
		super().__init__(N=N, **kwargs)
		self.wts = wts
		self.targets = targets
		# assert p in {1, 2}, f'p={p} is not supported'
		self.p = p


	@property
	def n_constraints(self):
		# braodcast wts with targets and count how many are non-zero
		return self.N - np.isclose(np.broadcast_to(self.wts, self.targets.shape), 0.).sum()


	def compute_loss(self, params):
		estimate = self.estimate(params)

		diffs = np.abs(estimate - self.targets) ** self.p

		return np.sum(self.wts * diffs)



class MarginalLoss(TargetLoss):
	def __init__(self, marginals, **kwargs):
		super().__init__(targets=marginals, **kwargs)


	def estimate(self, params):
		marginals = np.asarray([self.marginal(params, i) for i in range(self.N)])
		return marginals



class CorrelationLoss(TargetLoss):
	def __init__(self, correlations, **kwargs):
		super().__init__(targets=correlations, **kwargs)
		self.pairs = [(i, j) for i in range(self.N) for j in range(i+1, self.N)]


	def estimate(self, params):
		corrs = np.asarray([self.correlation(params, i, j) for i, j in self.pairs])
		return corrs



class FullySpecifiedLoss(TargetLoss):
	def __init__(self, moments=None, *, N=None, **kwargs):
		'''
		first N moments are marginals, then N(N-1)/2 covariances,
		and so on (so all higher order moments are included)
		'''
		if N is None:
			N = np.log2(len(moments)+1).round().astype(int)
		super().__init__(targets=moments, N=N, **kwargs)


	@staticmethod
	def max_entropy_moments(N: int):
		moments = np.zeros((2**N - 1,))
		moments[:N] = 0.5
		return moments


	@staticmethod
	def n_combos(indices, *, max_order=None): # not including empty set or full set
		max_order = max_order or len(indices)
		for r in range(1, max_order+1):
			combinations = itertools.combinations(indices, r)
			yield from combinations


	@classmethod
	def groups(cls, N, *, max_order=None):
		yield from cls.n_combos(list(range(N)), max_order=max_order)


	@classmethod
	def split_group(cls, base):
		total = set(base)
		for group in cls.n_combos(base):
			yield group, tuple(total.difference(group))


	@classmethod
	def estimate(cls, params):
		'''
		estimates all moments from params

		:param params:
		:return:
		'''




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
				* marginals[list(marginal_indices)].prod() #if len(marginal_indices) else 1. # first order moments
			for marginal_indices, moment_indices in all_splits(group)]

			n_body_correlations.append(sum(terms) / stds[list(group)].prod())

		return n_body_correlations



def generate_joint_bernoulli(marginals, correlations, *, x0=None, max_order=None):
	N = len(marginals)

	moments = Bernoulli_N_Body_Correlations.max_entropy_setting(N)
	moments[:N] = marginals
	moments[N:N+len(correlations)] = correlations

	wts = np.ones((2 ** N - 1,)) * 1.
	wts[len(marginals) : len(marginals) + len(correlations)] = 5.
	wts[len(marginals)+len(correlations):] = 1.

	return optimize_joint_bernoulli(moments, x0=x0, wts=wts, max_order=max_order)



def optimize_joint_bernoulli(moments, *, x0=None, wts=None, max_order=None, p=2.):
	# moments : (2**N-1,)
	N = int(np.round(np.log2(len(moments)+1)))
	
	marginals = moments[:N]
	assert np.all(marginals >= 0) and np.all(marginals <= 1), f'invalid first moments: {marginals}'
	assert np.all(np.abs(moments[N:]) <= 1), f'invalid higher order moments: {moments[N:]}'

	if x0 is None:
		x0 = np.zeros((2 ** N,))

	constraints = [int(special.comb(N, r)) for r in range(1, N+1)]
	if max_order is not None:
		constraints = constraints[:max_order]
	num_constraints = sum(constraints)

	if wts is None:
		wts = np.ones((2 ** N-1,))
	wts = np.asarray(wts)
	wts = wts[:num_constraints]

	targets = moments[:num_constraints].copy()

	system = Bernoulli_N_Body_Correlations(N, max_order=max_order)

	def step(params):
		probs = special.softmax(params)#.reshape([2] * N)
		estimate = system.compute(probs)
		estimate = np.asarray(estimate)

		error = np.abs(estimate - targets) ** p
		loss = wts @ error
		return loss

	sol = opt.minimize(step, x0)

	return special.softmax(sol.x).reshape(([2] * N))



# def test_optim_bernoulli():
# 	gen = np.random.RandomState()
#
# 	N = 5
#
# 	max_order = None
# 	# max_order = 2
#
# 	marginals = gen.uniform(0.1, 0.9, size=N)
# 	# marginals = np.ones(N) * 0.5
#
# 	correlations = prentice_bounds(marginals) \
# 	               * (2 * gen.uniform(size=N * (N - 1) // 2) - 1)
# 	# correlations = np.ones(N * (N - 1) // 2) * -0.9
# 	# correlations[0] = 0.2
# 	# correlations = np.abs(correlations)
#
# 	moments = Bernoulli_N_Body_Correlations.max_entropy_setting(N)
# 	moments[:N] = marginals
# 	moments[N:N+len(correlations)] = correlations
#
# 	params = generate_joint_bernoulli(marginals, correlations, max_order=max_order)
# 	# params = optimize_joint_bernoulli(moments, max_order=max_order)
#
# 	D = JointDistribution(params=params)
#
# 	mar = marginals
# 	amr = D.marginals()
#
# 	cor = correlations
# 	acr = D.corr()
#
# 	targets = \
# 		moments.tolist()
# 	final = \
# 		Bernoulli_N_Body_Correlations(N, max_order=max_order).compute(params.reshape(-1))
#
# 	samples = gen.choice(np.arange(2**N).astype(int), p=params.reshape(-1), size=50000)
# 	# counts = np.bincount(samples, minlength=2**N)
# 	# emp = counts / counts.sum()
#
# 	print('actual_corr',)
# 	print('actual_marginals')
#
# 	# assert isinstance(result, np.ndarray)
# 	# assert len(result) == 3
#
# 	pass
#
# 	# params = np.asarray([0.055, 0.00249, 0.04, 0.0025, 0.1866, 0.055853, 0.218348, 0.439147])
# 	# params /= params.sum()
# 	#
# 	# D = JointDistribution(params=params.reshape([2] * N))





#
# def generate_natural_bernoulli(marginals, correlations, seed=None):
# 	# https://pages.stat.wisc.edu/~wahba/ftp1/tr1171.pdf
#
#
# 	# step 1: moments -> natural parameters
#
#
#
#
# 	# step 2: natural parameters -> probs
#
# 	pass
#
#
#
#
#
# def test_natural_bernoulli():
# 	gen = np.random.RandomState()
#
# 	N = 4
#
# 	marginals = gen.uniform(0.1, 0.9, size=N)
# 	# marginals = np.ones(N) * 0.5
#
# 	correlations = prentice_bounds(marginals) \
# 	               * (2 * gen.uniform(size=N * (N - 1) // 2) - 1)
# 	# correlations = np.ones(N * (N - 1) // 2) * 0.
# 	# correlations[0] = 0.2
# 	correlations = np.abs(correlations)
#
# 	params = generate_natural_bernoulli(marginals, correlations)
#
# 	D = JointDistribution(params=params)
#
# 	actual_marginals = D.marginals()
# 	actual_corr = D.corr()
#
# 	print('actual_corr', actual_corr)
# 	print('actual_marginals', actual_marginals)
#
# 	# assert isinstance(result, np.ndarray)
# 	# assert len(result) == 3
#
# 	pass
#
# 	# params = np.asarray([0.055, 0.00249, 0.04, 0.0025, 0.1866, 0.055853, 0.218348, 0.439147])
# 	# params /= params.sum()
# 	#
# 	# D = JointDistribution(params=params.reshape([2] * N))









# import torch
# from torch import nn
# from torch.nn import functional as F

#
# def bernoulli_losses():
# 	pass
#
#
# class BernoulliLoss(nn.Module):
# 	def __init__(self, N, **kwargs):
# 		super().__init__(**kwargs)
# 		self.N = N
#
#
# 	@property
# 	def n_constraints(self):
# 		raise NotImplementedError
#
#
# 	def marginal(self, params, *indices: int, val: int = 1) -> float:
# 		sel = [slice(None)] * len(params.shape)
# 		for i in indices:
# 			sel[i] = val
# 		return params[sel].sum()
#
#
# 	def variance(self, params, i: int) -> float:
# 		marginal = self.marginal(params, i)
# 		return marginal * (1 - marginal)
#
#
# 	def covariance(self, params, i: int, j: int) -> float:
# 		marginal_i = self.marginal(params, i)
# 		marginal_j = self.marginal(params, j)
# 		joint = self.marginal(params, i, j)
# 		return joint - marginal_i * marginal_j
#
#
# 	def correlation(self, params, i: int, j: int) -> float:
# 		covariance = self.covariance(params, i, j)
# 		variance_i = self.variance(params, i)
# 		variance_j = self.variance(params, j)
# 		return covariance / (variance_i * variance_j).sqrt()
#
#
# 	def compute_loss(self, params):
# 		raise NotImplementedError
#
#
#
# class TargetLoss(BernoulliLoss):
# 	def __init__(self, targets, *, wts=None, p=2, N=None, **kwargs):
# 		if N is None:
# 			N = len(targets)
# 		if wts is None:
# 			wts = torch.ones(N)
# 		if not isinstance(wts, torch.Tensor):
# 			wts = torch.as_tensor(wts).float()
# 		if not isinstance(targets, torch.Tensor):
# 			targets = torch.as_tensor(targets).float()
# 		super().__init__(N=N, **kwargs)
# 		self.wts = wts
# 		self.targets = targets
# 		assert p in {1, 2}, f'p={p} is not supported'
# 		self.criterion = nn.MSELoss(reduction='none') if p == 2 else nn.L1Loss(reduction='none')
#
#
# 	@property
# 	def n_constraints(self):
# 		# braodcast wts with targets and count how many are non-zero
# 		return self.N - torch.broadcast_to(self.wts, self.targets.shape).isclose(torch.tensor(0.)).sum().item()
#
#
# 	def compute_loss(self, params):
# 		estimate = self(params)
# 		losses = self.criterion(estimate, self.targets)#.sum(dim=1)
# 		return losses.mul(self.wts).sum()
#
#
#
# class MarginalLoss(TargetLoss):
# 	def __init__(self, marginals, **kwargs):
# 		super().__init__(targets=marginals, **kwargs)
#
#
# 	def forward(self, params):
# 		marginals = torch.stack([self.marginal(params, i) for i in range(self.N)])
# 		return marginals
#
#
#
# class CorrelationLoss(TargetLoss):
# 	def __init__(self, correlations, **kwargs):
# 		super().__init__(targets=correlations, **kwargs)
# 		self.pairs = [(i, j) for i in range(self.N) for j in range(i+1, self.N)]
#
#
# 	def forward(self, params):
# 		corrs = torch.stack([self.correlation(params, i, j) for i, j in self.pairs])
# 		return corrs
#
#
#
# def optimize_params(criterion, x0, *, maxiter=1000, threshold=1e-5):
# 	x = x0.clone()
# 	x.requires_grad_(True)
#
# 	optim = torch.optim.Adam([x], lr=1e-2)
#
# 	prev = float('inf')
# 	i = 0
# 	for i in range(1, maxiter+1):
# 		optim.zero_grad()
# 		loss = criterion(F.softmax(x))
# 		loss.backward()
# 		optim.step()
#
# 		if torch.abs(loss - prev) < threshold:
# 			break
# 		prev = loss.item()
#
# 	return i, x
#
#
#
# def optimize_joint_bernoulli(marginals, correlations, *, x0=None,
#                        seed=None, eps=1e-3, maxiter=1000, losses=None):
# 	# params : (2**N,)
# 	# marginals : (N,)
# 	# correlations : (N * (N - 1) // 2,)
#
# 	if losses is None:
# 		losses = []
#
# 	N = len(marginals)
#
# 	lim = prentice_bounds(marginals)
# 	accept = np.all(np.abs(correlations) <= lim)
#
# 	if x0 is None:
# 		x0 = torch.zeros(2 ** N)
#
# 	losses = [MarginalLoss(marginals), CorrelationLoss(correlations, wts=10), *losses]
#
# 	constraints = sum(loss.n_constraints for loss in losses)
# 	dof = 2 ** N - constraints
#
#
# 	def criterion(params):
# 		params = params.view(*([2] * N))
# 		vals = [loss.compute_loss(params) for loss in losses]
# 		return sum(vals)
#
# 	i, x = optimize_params(criterion, x0, maxiter=maxiter, threshold=eps)
#
# 	return F.softmax(x).detach().numpy().reshape([2] * N)
#
#
#
# def test_optim_bernoulli():
# 	gen = np.random.RandomState()
#
# 	N = 4
#
# 	marginals = gen.uniform(0.1, 0.9, size=N)
# 	# marginals = np.ones(N) * 0.5
#
# 	correlations = prentice_bounds(marginals) \
# 	               * (2 * gen.uniform(size=N * (N - 1) // 2) - 1)
# 	# correlations = np.ones(N * (N - 1) // 2) * 0.
# 	# correlations[0] = 0.2
# 	correlations = np.abs(correlations)
#
# 	params = optimize_joint_bernoulli(marginals, correlations)
#
# 	D = JointDistribution(params=params)
#
# 	actual_marginals = D.marginals()
# 	actual_corr = D.corr()
#
# 	print('actual_corr', actual_corr)
# 	print('actual_marginals', actual_marginals)
#
# 	# assert isinstance(result, np.ndarray)
# 	# assert len(result) == 3
#
# 	pass
#
# 	# params = np.asarray([0.055, 0.00249, 0.04, 0.0025, 0.1866, 0.055853, 0.218348, 0.439147])
# 	# params /= params.sum()
# 	#
# 	# D = JointDistribution(params=params.reshape([2] * N))



########################################################################################################################


def gaussian_cdf(x, *, mu=None, sigma=None):
	if mu is None and sigma is None:
		return stats.norm.cdf(x)
	return 0.5 * (1 + special.erf((x - mu) / (sigma * np.sqrt(2))))


def gaussian_inv_cdf(p, *, mu=None, sigma=None):
	if mu is None and sigma is None:
		return stats.norm.ppf(p)
	return mu + sigma * np.sqrt(2) * special.erfinv(2 * p - 1)


def gaussian_2d_cdf(x, y, rho=0):
	cov = np.asarray([[1, rho], [rho, 1]])
	mvn = stats.multivariate_normal(mean=np.zeros((2,)), cov=cov)
	return mvn.cdf([x, y])


# def gaussian_2d_inv_cdf(p, *, rho=0, mu=None, sigma=None):
# 	cov = np.asarray([[1, rho], [rho, 1]]) if sigma is None else sigma
# 	if mu is None:
# 		mu = np.zeros((2,))
# 	mvn = stats.multivariate_normal(mean=mu, cov=cov)
# 	return mvn.ppf(p)


def find_rho(target, x, y, *, eps=1e-3):
	# loc : (N, 2)
	return opt.brentq(lambda rho: gaussian_2d_cdf(x, y, rho=rho) - target, -1+eps, 1-eps)


def generate_gaussian_bernoulli(marginals,
                                correlations, *,
                                nsamples=1000, seed=None):
	
	N = len(marginals)
	if N > 10:
		print(f'WARNING: generating joint distribution with {N} variables (generally not recommended for N > 10)')

	assert len(correlations) == N * (N - 1) // 2, f'correlations = {correlations} should have length {N * (N - 1) // 2}'

	assert all(m is None or 0 <= m <= 1 for m in marginals), f'marginals = {marginals} should be in [0, 1]'
	assert all(c is None or -1 <= c <= 1 for c in correlations), f'correlations = {correlations} should be in [-1, 1]'

	gen = np.random.RandomState(seed)

	# check Prentice constraints

	lim = prentice_bounds(marginals)
	accept = np.all(np.abs(correlations) <= lim)

	b_corr = squareform(correlations) + np.eye(N)
	
	b_cov = b_corr * np.outer(marginals, marginals)

	i, j = np.triu_indices(N, k=1)
	targets = b_cov[i, j]
	baselines = marginals[i] * marginals[j]
	
	mu = gaussian_inv_cdf(marginals)
	
	rhos = []
	
	for t, mui, muj, b in zip(targets, mu[i], mu[j], baselines):
		# b2 = gaussian_cdf(mui) * gaussian_cdf(muj)
		rho = find_rho(t+b, mui, muj)
		rhos.append(rho)
	
	rho = np.asarray(rhos)
	
	# rho = find_rho(targets, np.stack([mu[i], mu[j]], axis=-1), baseline=marginals[i] * marginals[j])
	
	diag2 = gaussian_cdf(mu) * gaussian_cdf(-mu)
	diag = marginals * (1 - marginals)
	
	sigma = squareform(rho) + np.diag(diag)
	
	# generate samples
	
	samples = gen.multivariate_normal(mean=mu, cov=sigma, size=nsamples)
	
	# convert to Bernoulli
	
	samples = (samples > 0).astype(float)
	
	bits = 2 ** np.arange(N-1, -1, -1)

	counts = np.bincount(np.dot(samples, bits).astype(int), minlength=2**N)

	params = counts / counts.sum()
	
	return params.reshape([2]*N)


# def test_gaussian_bernoulli():
#
# 	gen = np.random.RandomState()
#
# 	N = 3
#
# 	marginals = gen.uniform(0.1, 0.9, size=N)
# 	# marginals = np.ones(N) * 0.5
#
# 	correlations = prentice_bounds(marginals) * (2 * gen.uniform(size=N * (N - 1) // 2) - 1)
# 	# correlations = np.ones(N * (N - 1) // 2) * 0.
# 	# correlations[0] = 0.2
# 	correlations = np.abs(correlations)
# 	# correlations = np.asarray([0.635244, -0.70220, -0.4590])
#
# 	params = generate_gaussian_bernoulli(marginals, correlations, nsamples=50000)
#
# 	D = JointDistribution(params=params)
#
# 	# params = np.asarray([0.055, 0.00249, 0.04, 0.0025, 0.1866, 0.055853, 0.218348, 0.439147])
# 	# params /= params.sum()
# 	#
# 	# D = JointDistribution(params=params.reshape([2] * N))
#
# 	actual_corr = D.corr()
# 	actual_marginals = D.marginals()
#
# 	print('actual_corr', actual_corr)
# 	print('actual_marginals', actual_marginals)
#
# 	# assert isinstance(result, np.ndarray)
# 	# assert len(result) == 3
#
# 	pass

	

# def test_cdf():
#
# 	# x = np.linspace(-3, 3, 100)
# 	#
# 	# y = gaussian_cdf(x)
# 	#
# 	import matplotlib.pyplot as plt
# 	# plt.plot(x, y)
# 	# plt.show()
#
# 	# x = np.linspace(-3, 3, 100)
# 	# y = np.linspace(-3, 3, 100)
# 	# X, Y = np.meshgrid(x, y)
# 	# points = np.column_stack([X.flatten(), Y.flatten()])
#
# 	points = np.zeros((100, 2))
#
# 	x = np.linspace(-0.2, 0.2, 100)
# 	cdf = gaussian_2d_cdf(points, rho=x)
# 	plt.plot(x, cdf)
#
# 	# CDF = cdf.reshape(X.shape)
#
# 	# # plot cdf as heatmap
# 	# plt.imshow(CDF, extent=[-3, 3, -3, 3])
# 	# plt.colorbar()
# 	#
# 	plt.show()
	





import numpy as np
from math import sqrt
from warnings import warn



def cBernMdepk_single(m, p, rho, k):
	p += [p[m - 1]] * k
	for w in range(k):
		rho[w][(m - w):m] = [0] * (m - w)
	
	Y = np.zeros((k, m))
	b = np.zeros((k, m))
	for w in range(k):
		for j in range(m):
			rwj = rho[w][j]
			b[w, j] = p[j] * p[w + j] / (
						rwj * sqrt(p[j] * p[w + j] * (1 - p[j]) * (1 - p[w + j])) + p[j] * p[w + j])
			Y[w, j] = np.random.binomial(1, b[w, j])
	
	a = np.zeros(m)
	U = np.zeros(m)
	X = np.zeros(m)
	for w in range(m):
		prod1 = prod2 = 1
		if w == 0:
			for l in range(k):
				prod1 *= b[l, w]
				prod2 *= Y[l, w]
		elif w < k:
			for l in range(k):
				prod1 *= b[l, w]
				prod2 *= Y[l, w]
			for l in range(w):
				prod1 *= b[l, w - l]
				prod2 *= Y[l, w - l]
		else:
			for l in range(k):
				prod1 *= b[l, w] * b[l, w - l]
				prod2 *= Y[l, w] * Y[l, w - l]
		
		a[w] = p[w] / prod1
		U[w] = np.random.binomial(1, a[w])
		X[w] = U[w] * prod2
	
	return X



def cBernMdepk(n, p, rho, k):
	m = len(p)
	# X = cBernMdepk_single(m, p.copy(), [r.copy() for r in rho], k)
	#
	# if np.isnan(X).any():
	# 	warn("Invalid Input: Please adjust the input parameters.")
	# else:
	simX = np.array([cBernMdepk_single(m, p.copy(), [r.copy() for r in rho], k) for _ in range(n)])
	return simX



# def test_cBernMdepk_single():
# 	# gen = np.random.RandomState()
# 	#
# 	N = 3
# 	#
# 	# marginals = gen.uniform(0.1, 0.9, size=N)
# 	# marginals = np.ones(N) * 0.5
# 	#
# 	# correlations = prentice_bounds(marginals) * (2 * gen.uniform(size=N * (N - 1) // 2) - 1)
# 	# correlations = np.ones(N * (N - 1) // 2) * 0.
# 	# correlations[0] = 0.2
# 	# # correlations = np.asarray([0.635244, -0.70220, -0.4590])
# 	#
# 	# p, rho = marginals, squareform(correlations)+0*np.eye(N)
# 	# p = p.tolist()
# 	# rho = rho.tolist()
# 	#
# 	# result = cBernMdepk(10000, p, rho, 2)
# 	# samples = result.astype(float)
# 	#
# 	# bits = 2 ** np.arange(N-1, -1, -1)
# 	#
# 	# counts = np.bincount(np.dot(samples, bits).astype(int), minlength=2**N)
# 	#
# 	# params = counts / counts.sum()
# 	#
# 	# D = JointDistribution(params=params.reshape([2]*N))
#
# 	params = np.asarray([0.055, 0.00249, 0.04, 0.0025, 0.1866, 0.055853, 0.218348, 0.439147])
# 	params /= params.sum()
#
# 	D = JointDistribution(params=params.reshape([2]*N))
#
# 	actual_corr = D.corr()
# 	actual_marginals = D.marginals()
#
# 	print('actual_corr', actual_corr)
# 	print('actual_marginals', actual_marginals)
#
# 	# assert isinstance(result, np.ndarray)
# 	# assert len(result) == 3



# def test_cBernMdepk():
# 	result = cBernMdepk(1000, [0.5, 0.7, 0.2], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 2)
# 	total = result.sum(0)
# 	assert isinstance(result, np.ndarray)




# def test_cBernMdepk_nonzero_correlation():
# 	np.random.seed(0)  # Set seed for reproducibility
# 	# p = [0.5, 0.7, 0.2]
# 	p = [0.5, 0.7, 0.2]
# 	rho = np.zeros((3, 3))
# 	rho[0,0] = 0.5
# 	rho = rho.tolist()
# 	# rho = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
# 	result = cBernMdepk(2, p, rho, 2)
#
# 	expected_result = np.array([[1., 0., 0.], [0., 0., 0.]])
#
# 	assert np.allclose(result, expected_result, atol=1e-6), f"Expected {expected_result} but got {result}"







import numpy as np
from scipy.stats import binom


# def cBernMdepk_single(p, rho):
# 	m = len(p)
# 	k = len(p)
#
# 	p = np.pad(p, (0, k), 'constant', constant_values=0)
# 	for w in range(k):
# 		p[m + w] = p[m]
# 		rho[w][(m-w):m] = 0
#
# 	Y = b = np.zeros((k, m))
# 	for w in range(k):
# 		for j in range(m):
# 			b[w, j] = p[j]*p[w+j] / (rho[w][j]*np.sqrt(p[j]*p[w+j]*(1-p[j])*(1-p[w+j])) + p[j]*p[w+j])
# 			Y[w, j] = np.random.uniform() < b[w, j]
# 			# Y[w, j] = binom.rvs(1, b[w, j])
#
# 	a = U = X = np.zeros(m)
# 	for w in range(m):
# 		prod1 = prod2 = 1
# 		if w == 0:
# 			for l in range(k):
# 				prod1 *= b[l, w]
# 				prod2 *= Y[l, w]
# 		elif w <= k:
# 			for l in range(k):
# 				prod1 *= b[l, w]
# 				prod2 *= Y[l, w]
# 			for l in range(w):
# 				prod1 *= b[l, w-l]
# 				prod2 *= Y[l, w-l]
# 		else:
# 			for l in range(k):
# 				prod1 *= b[l, w]*b[l, w-l]
# 				prod2 *= Y[l, w]*Y[l, w-l]
#
# 		a[w] = p[w] / prod1
# 		U[w] = np.random.uniform() < a[w]
# 		# U[w] = binom.rvs(1, a[w])
# 		X[w] = U[w] * prod2
#
# 	return X



# def test_generate_joint():
#
# 	gen = np.random.RandomState()
#
# 	N = 3
#
# 	marginals = gen.uniform(0.1, 0.9, size=N)
# 	marginals = np.ones(N) * 0.5
#
# 	correlations = prentice_bounds(marginals) * (2 * gen.uniform(size=N * (N - 1) // 2) - 1)
# 	correlations = np.ones(N * (N - 1) // 2) * 0.
# 	# correlations = np.asarray([0.635244, -0.70220, -0.4590])
#
# 	p, rho = marginals, squareform(correlations)+np.eye(N)
#
# 	samples = np.stack([cBernMdepk_single(p, rho) for _ in range(1000)])
# 	samples = samples.astype(float)
#
# 	bits = 2 ** np.arange(N-1, -1, -1)
#
# 	counts = np.bincount(np.dot(samples, bits).astype(int), minlength=2**N)
#
# 	params = counts / counts.sum()
#
#
#
#
#
# 	print(f'failed {failed/1000:.1%} / 100')





def generate_joint_jiang(marginals: Sequence[float],
						 correlations: Sequence[float],
						 *, nsamples=10, seed=None):
	'''
	implemented based on https://arxiv.org/pdf/2007.14080.pdf

	apparently an N dimensional bernoulli can be transformed into an N+1 dimensional symmetric bernoulli
	(ie. marginals are all 1/2)

	:param marginals: marginal prob of each var numpy array shape (3,)
	:param correlations: correlations between each pair of vars (in condensed form [(0,1), (0,2), (1,2)])
	numpy array shape (3,)

	any of the above can be None, in which case it is sampled from a uniform distribution

	:return: bernoulli joint distribution numpy array shape (2, 2, 2)
	'''

	N = len(marginals)
	if N > 10:
		print(f'WARNING: generating joint distribution with {N} variables (generally not recommended for N > 10)')

	assert len(correlations) == N * (N - 1) // 2, f'correlations = {correlations} should have length {N * (N - 1) // 2}'

	assert all(m is None or 0 <= m <= 1 for m in marginals), f'marginals = {marginals} should be in [0, 1]'
	assert all(c is None or -1 <= c <= 1 for c in correlations), f'correlations = {correlations} should be in [-1, 1]'

	gen = np.random.RandomState(seed)

	# check Prentice constraints

	lim = prentice_bounds(marginals)
	accept = np.all(np.abs(correlations) <= lim)


	rho = squareform(correlations) + np.eye(N)

	beta = np.zeros((N, N))

	for i, j in np.ndindex(N, N):

		p_j = marginals[j]
		p_ij = marginals[min(N-1, i+j)]

		num = p_j * p_ij
		den = num + rho[i, j] * np.sqrt(num * (1 - p_j) * (1 - p_ij))

		beta[i, j] = num / den

	Y = gen.uniform(0, 1, size=(N*N, nsamples)) - beta.reshape(-1, 1) < 0
	Y = Y.astype(int).reshape(N, N, nsamples)


	alpha = marginals[0] / np.prod(beta[:, 0])
	fail = alpha > 1

	U = gen.uniform(0, 1, size=(nsamples,)) - alpha < 0
	U = U.astype(int)

	Xs = [U * np.prod(Y[:, 0, :], axis=0)]

	for i in range(1, N):
		sel = np.eye(i, dtype=bool)[::-1]
		bsel = beta[:i, :i][sel]

		alpha = marginals[i] / np.prod(beta[:, i])
		alpha *= np.prod(bsel)

		fail = fail | (alpha > 1)

		U = gen.uniform(0, 1, size=(nsamples,)) - alpha < 0
		U = U.astype(int)

		Xi = U * np.prod(Y[:i, i, :], axis=0)
		Xi *= np.prod(Y[:i, :i, :][sel, :], axis=0)

		Xs.append(Xi)

	X = np.stack(Xs, axis=0)

	print(X)

	return X



# def test_generate_joint():
#
# 	gen = np.random.RandomState()
#
# 	N = 3
#
# 	marginals = gen.uniform(0.1, 0.9, size=N)
# 	marginals = np.ones(N) * 0.5
#
# 	correlations = np.ones(N * (N - 1) // 2) * 0.
# 	# correlations = np.asarray([0.635244, -0.70220, -0.45903])
#
# 	params = generate_joint_jiang(marginals, correlations)
#
# 	if np.allclose(params.sum(), 1) and np.all(params >= 0):
# 		pass
#
# 		D = JointDistribution(params=params)
#
# 		actual_corr = D.corr()
# 		actual_marginals = D.marginals()
#
# 		print('actual_corr', actual_corr)
# 		print('actual_marginals', actual_marginals)
#
# 	else:
# 		failed += 1
#
# 	print(f'failed {failed/1000:.1%} / 100')





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
			return np.asarray([self.marginal(i, val=val) for i in range(self.N)])
		assert len(conds) == self.N, f'Expected {self.N} values, got {len(conds)}'
		condition = self.prob(*conds)
		return np.asarray([self.marginal(i, val=val) / condition if c is None else c for i, c in enumerate(conds)])


	def marginal(self, *indices: int, val: int = 1) -> float:
		sel = [slice(None)] * self.N
		for i in indices:
			sel[i] = val
		return self._params[tuple(sel)].sum()


	def variance(self, i: int) -> float:
		marginal = self.marginal(i)
		return marginal * (1 - marginal)


	def covariance(self, i: int, j: int) -> float:
		marginal_i = self.marginal(i)
		marginal_j = self.marginal(j)
		joint = self.marginal(i, j)
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




@lru_cache(maxsize=1000)
def _get_constraints(N: int):
	corners = np.asarray([[0, *inds] for inds in np.ndindex(*[2] * N)]).T
	i, j = np.triu_indices(N+1, k=1)
	constraints = (-1.) ** (corners[i] + corners[j])
	constraints = np.concatenate([constraints, np.ones((1, 2**N))]) # constraint that sum of all params = 1
	return constraints



def generate_joint_huber(marginals: Sequence[float],
							   correlations: Sequence[float],
							   ):
	'''
	implemented based on https://jsdajournal.springeropen.com/articles/10.1186/s40488-019-0091-5

	apparently an N dimensional bernoulli can be transformed into an N+1 dimensional symmetric bernoulli
	(ie. marginals are all 1/2)

	:param marginals: marginal prob of each var numpy array shape (3,)
	:param correlations: correlations between each pair of vars (in condensed form [(0,1), (0,2), (1,2)])
	numpy array shape (3,)

	any of the above can be None, in which case it is sampled from a uniform distribution

	:return: bernoulli joint distribution numpy array shape (2, 2, 2)
	'''

	N = len(marginals)
	if N > 10:
		print(f'WARNING: generating joint distribution with {N} variables (generally not recommended for N > 10)')

	assert len(correlations) == N * (N - 1) // 2, f'correlations = {correlations} should have length {N * (N - 1) // 2}'

	assert all(m is None or 0 <= m <= 1 for m in marginals), f'marginals = {marginals} should be in [0, 1]'
	assert all(c is None or -1 <= c <= 1 for c in correlations), f'correlations = {correlations} should be in [-1, 1]'

	targets = np.concatenate([marginals, correlations, [1]])
	N, targets = 2, np.asarray([0.635244, -0.70220, -0.45903, 1.]) # example from paper

	constraints = _get_constraints(N)

	sol, r = opt.nnls(constraints, targets)

	total = sol.sum()

	# if not np.isclose(sol.sum(), 1) or np.any(sol < 0) or np.any(sol > 1):
	# 	raise ValueError(f'sol = {sol} should be in [0, 1]')

	corners = np.asarray([[0, *inds] for inds in np.ndindex(*[2] * N)])

	params = np.concatenate([sol/2]*2).reshape([2] * (N+1))


	dis = JointDistribution(params=params)

	mar = dis.marginals()
	cor = dis.corr()


	return sol.reshape([2] * (N))



# def test_generate_joint():
#
# 	gen = np.random.RandomState()
#
# 	N = 3
#
# 	failed = 0
# 	for _ in range(1000):
#
# 		marginals = gen.uniform(0.1, 0.9, size=N)
# 		marginals = np.ones(N) * 0.5
#
# 		correlations = gen.uniform(0.1, 0.9, size=N * (N - 1) // 2)
# 		i, j = np.triu_indices(N, k=1)
# 		mi, mj = marginals[i], marginals[j]
# 		min_corr = -np.minimum(np.minimum(np.minimum((1-mi)/mi, mi/(1-mi)), (1-mj)/mj), mj/(1-mj))
# 		correlations = (1-min_corr) * correlations + min_corr
#
# 		# correlations = np.ones(N * (N - 1) // 2) * 0.
# 		correlations = np.asarray([0.635244, -0.70220, -0.45903])
#
# 		params = generate_joint(marginals, correlations)
#
# 		if np.allclose(params.sum(), 1) and np.all(params >= 0):
# 			pass
#
# 			D = JointDistribution(params=params)
#
# 			actual_corr = D.corr()
# 			actual_marginals = D.marginals()
#
# 			print('actual_corr', actual_corr)
# 			print('actual_marginals', actual_marginals)
#
# 		else:
# 			failed += 1
#
# 	print(f'failed {failed/1000:.1%} / 100')






def generate_joint_numeric(N=None, *, marginals: Optional[Sequence[Optional[float]]] = None,
						   correlations: Optional[Sequence[Optional[float]]] = None,
						   range_marginal=0.98, range_correlation=0.98,
						   seed=None, nsamples=10000):
	'''
	:param marginals: marginal prob of each var numpy array shape (3,)
	:param correlations: correlations between each pair of vars (in order of var1, var2, var3)
	:param range_marginal: range of marginal prob
	:param range_correlation: range of correlation
	:param alpha_choice: function to choose alpha
	:param seed: random seed
	:param fuel: number of attempts to generate a valid joint distribution
	:return:
	'''

	if N is None:
		assert marginals is not None or correlations is not None, 'Either N or marginals and correlations must be specified'
		if marginals is not None:
			N = len(marginals)
		else:
			N = int((np.sqrt(8 * len(correlations) + 1) + 1) / 2) #int(np.sqrt(len(correlations) * 2))

	gen = np.random.RandomState(seed)

	if marginals is None:
		marginals = N*[None]
	if correlations is None:
		correlations = (N-1)*N//2*[None]

	marginals = np.asarray([gen.uniform((1-range_marginal) / 2, 1 - (1-range_marginal)/2) if m is None else m for m in marginals])
	correlations = np.asarray([gen.uniform(-range_correlation, range_correlation) if c is None else c for c in correlations])

	assert all(m is None or 0 <= m <= 1 for m in marginals), f'marginals = {marginals} should be in [0, 1]'
	assert all(c is None or -1 <= c <= 1 for c in correlations), f'correlations = {correlations} should be in [-1, 1]'

	corr = squareform(correlations) + np.eye(N)
	std = np.sqrt(marginals * (1 - marginals))

	cov = corr * np.outer(std, std)

	# generate gaussian samples
	samples = gen.multivariate_normal(np.zeros(N), cov, size=nsamples)

	def sigmoid(x):
		return 1.0 / (1.0 + np.exp(-x))
	# convert to bernoulli
	samples = (sigmoid(samples) - marginals.reshape(1, -1) < 0).astype(int)

	bits = 2 ** np.arange(N-1, -1, -1)

	counts = np.bincount(np.dot(samples, bits), minlength=2**N)

	params = counts / counts.sum()
	return params.reshape((2,)*N)



# def test_sample_numeric_joint():
# 	# print(generate_joint_numeric(correlations=[0]*3, marginals=[0.5]*3, seed=101))
# 	#
# 	# print(generate_joint_numeric(correlations=[1]*3, marginals=[0.5]*3, seed=101))
# 	# print(generate_joint_numeric(correlations=[-1, 0, 0], marginals=[0.5]*3, seed=101))
# 	#
# 	# print(generate_joint_numeric(correlations=[0]*3, marginals=[0.5]*3, seed=101))
#
# 	# print(generate_joint_numeric(correlations=[0, 0, 0], seed=101, range_marginal=0.5, range_correlation=0.5))
# 	# print(generate_joint_numeric(marginals=[0.55, 0.5, 0.5], seed=101, range_marginal=0.5, range_correlation=0.5))
#
# 	print(generate_joint_numeric(4, seed=101))
#
# 	for _ in range(100):
# 		print(generate_joint_numeric(3, seed=101))



# def test_verify_numeric_joint():
#
# 	# gen = np.random.RandomState(101)
# 	gen = np.random.RandomState()
#
# 	for _ in range(10):
#
# 		corr = gen.uniform(-1, 1, size=(3,))
# 		marginals = gen.uniform(0, 1, size=(3,))
#
# 		params = generate_joint_numeric(correlations=corr, marginals=marginals, seed=101)
#
# 		D = JointDistribution(params=params)
#
# 		actual_corr = D.corr()
# 		actual_marginals = D.marginals()
#
# 		print(corr, actual_corr)
#
# 		# assert np.allclose(actual_corr, corr), f'Expected {corr}, got {actual_corr}'
# 		# assert np.allclose(actual_marginals, marginals), f'Expected {marginals}, got {actual_marginals}'



def _convexity_from_correlation(c12, m1, m2):
	# return m1 * m2 + (1 - m1) * (1 - m2) + c12 * np.sqrt(m1 * m2 * ((1 - m1) * (1 - m2) + m1 * m2))
	return m1 * m2 + c12 * np.sqrt(m1 * m2 * (1-m1) * (1-m2)) + (1-m1) * (1-m2) + c12 * np.sqrt((1-m1) * (1-m2) * m1 * m2)


def generate_joint_3_variables(*, marginals: Optional[Sequence[Optional[float]]] = None,
							   correlations: Optional[Sequence[Optional[float]]] = None,
							   range_marginal=0.98, range_correlation=0.98,
							   alpha_choice=None,
							   seed=None, fuel=100):
	'''
	implemented based on https://arxiv.org/pdf/1311.2002.pdf

	apparently an N dimensional bernoulli can be transformed into an N+1 dimensional symmetric bernoulli
	(ie. marginals are all 1/2)

	:param marginals: marginal prob of each var numpy array shape (3,)
	:param correlations: correlations between each pair of vars (in condensed form [(0,1), (0,2), (1,2)])
	numpy array shape (3,)

	any of the above can be None, in which case it is sampled from a uniform distribution

	:return: bernoulli joint distribution numpy array shape (2, 2, 2)
	'''

	start_fuel = fuel
	gen = np.random.RandomState(seed)

	if marginals is None:
		marginals = [None, None, None]
	if correlations is None:
		correlations = [None, None, None]

	assert len(marginals) == 3, f'marginals = {marginals} should have length 3'
	assert len(correlations) == 3, f'correlations = {correlations} should have length 3'

	assert all(m is None or 0 <= m <= 1 for m in marginals), f'marginals = {marginals} should be in [0, 1]'
	assert all(c is None or -1 <= c <= 1 for c in correlations), f'correlations = {correlations} should be in [-1, 1]'

	while fuel > 0:
		mars = np.asarray([gen.uniform((1-range_marginal) / 2, 1 - (1-range_marginal)/2) if m is None else m for m in marginals])
		corrs = np.asarray([gen.uniform(-range_correlation, range_correlation) if c is None else c for c in correlations])

		# compute convexity parameters from correlations (for a symmetric bernoulli distribution) -> ??
		c12, c13, c23 = corrs
		m1, m2, m3 = mars

		l12 = _convexity_from_correlation(c12, m1, m2)
		l13 = _convexity_from_correlation(c13, m1, m3)
		l23 = _convexity_from_correlation(c23, m2, m3)
		l14, l24, l34 = m1, m2, m3

		l = max(l14 + l24 + l13 + l23,
				l14 + l34 + l12 + l23,
				l24 + l34 + l12 + l13)

		u = min(l12 + l23 + l13,
				l12 + l24 + l14,
				l13 + l34 + l14,
				l23 + l34 + l24)

		check1 = u >= 1
		check2 = l <= u + 1
		# assert u >= 1, f'u = {u} should be >= 1'
		# assert l <= u + 1, f'l = {l} should be <= u + 1 = {u + 1}'

		if not (check1 and check2):
			fuel -= 1
			continue

		# req: alpha in [l/2 - 1, (u - 1)/2] ^ [0, 1]
		lims = max(l/2 - 1, 0), min((u - 1)/2, 1)
		alpha = gen.uniform(*lims) if alpha_choice is None else alpha_choice*(lims[1] - lims[0]) + lims[0]

		q = np.array([
			0.5 * (l12 + l13 + l23) - 0.5 - alpha,
			-0.5 * (l14 + l24 + l13 + l23) + 1 + alpha,
			-0.5 * (l14 + l34 + l12 + l23) + 1 + alpha,
			0.5 * (l24 + l34 + l23) - 0.5 - alpha,
			-0.5 * (l24 + l34 + l12 + l13) + 1 + alpha,
			0.5 * (l14 + l34 + l13) - 0.5 - alpha,
			0.5 * (l34 + l24 + l12) - 0.5 - alpha,
			alpha,
		])

		_sum = q.sum()
		if np.all(q >= 0) and np.isclose(q.sum(), 1):
			break

		fuel -= 1

	else:
		raise RuntimeError(f'could not generate a valid bernoulli distribution (after {start_fuel} tries): '
						   f'marginals={marginals}, correlations={correlations}')

	return q.reshape((2, 2, 2))




# def test_sample_controlled_joint():
# 	# print(generate_joint_3_variables(correlations=[0]*3, marginals=[0.5]*3, alpha_choice=0., seed=101))
# 	# print(generate_joint_3_variables(correlations=[0]*3, marginals=[0.5]*3, alpha_choice=1., seed=101))
#
# 	# print(generate_joint_3_variables(correlations=[1]*3, marginals=[0.5]*3, seed=101))
# 	# print(generate_joint_3_variables(correlations=[-1, 0, 0], marginals=[0.5]*3, seed=101))
#
# 	# print(generate_joint_3_variables(correlations=[0]*3, marginals=[0.5]*3, seed=101))
#
# 	print(generate_joint_3_variables(correlations=[0, 0, 0], marginals=[0.5, 0.5, 0.5],
# 	                                 alpha_choice=1.,
# 	                                 seed=101, range_marginal=0.5, range_correlation=0.5))
#
# 	for _ in range(10):
# 		print(generate_joint_3_variables(correlations=[0, 0, 0], marginals=[0.55, 0.5, 0.5],
# 		                                 alpha_choice=1.,
# 		                                 seed=101, range_marginal=0.5, range_correlation=0.5))
#
#
# 	print(generate_joint_3_variables(correlations=[0, 0, 0], seed=101, range_marginal=0.5, range_correlation=0.5))
#
#
# 	for _ in range(100):
# 		print(generate_joint_3_variables(seed=101))
#
# 	pass






























