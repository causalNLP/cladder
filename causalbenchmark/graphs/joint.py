from typing import Sequence, Optional
import numpy as np



def _convexity_from_correlation(c12, m1, m2):
	# return m1 * m2 + (1 - m1) * (1 - m2) + c12 * np.sqrt(m1 * m2 * ((1 - m1) * (1 - m2) + m1 * m2))
	return m1 * m2 + c12 * np.sqrt(m1 * m2 * (1-m1) * (1-m2)) + (1-m1) * (1-m2) + c12 * np.sqrt((1-m1) * (1-m2) * m1 * m2)


def generate_joint_3_variables(*, marginals: Optional[Sequence[Optional[float]]] = None,
                               correlations: Optional[Sequence[Optional[float]]] = None,
                               range_marginal=0.98, range_correlation=0.02,
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








def test_sample_correlations():
	# print(generate_joint_3_variables(correlations=[0]*3, marginals=[0.5]*3, alpha_choice=0., seed=101))
	# print(generate_joint_3_variables(correlations=[0]*3, marginals=[0.5]*3, alpha_choice=1., seed=101))

	# print(generate_joint_3_variables(correlations=[1]*3, marginals=[0.5]*3, seed=101))
	# print(generate_joint_3_variables(correlations=[-1, 0, 0], marginals=[0.5]*3, seed=101))

	# print(generate_joint_3_variables(correlations=[0]*3, marginals=[0.5]*3, seed=101))

	print(generate_joint_3_variables(correlations=[0, 0, 0], marginals=[0.5, 0.5, 0.5],
	                                 alpha_choice=1.,
	                                 seed=101, range_marginal=0.5, range_correlation=0.5))

	for _ in range(10):
		print(generate_joint_3_variables(correlations=[0, 0, 0], marginals=[0.55, 0.5, 0.5],
		                                 alpha_choice=1.,
		                                 seed=101, range_marginal=0.5, range_correlation=0.5))


	print(generate_joint_3_variables(correlations=[0, 0, 0], seed=101, range_marginal=0.5, range_correlation=0.5))


	for _ in range(100):
		print(generate_joint_3_variables(seed=101))

	pass






























