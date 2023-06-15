
import numpy as np
from scipy import optimize as opt

from .base import create_graph


class OptimProblem:
	treatment = 'X'
	outcome = 'Y'

	def extract(self, graph):
		constraint = self.constraints(graph)

		x0 = (constraint.ub - constraint.lb) / 2 + constraint.lb
		return x0


	def constraints(self, graph):
		raise NotImplementedError


	def estimate(self, x):
		raise NotImplementedError


	def to_targets(self, x):
		raise NotImplementedError



class Confounding_ATE(OptimProblem):
	confounder = 'V1'


	# def extract(self, graph):
	# 	params = []
	#
	# 	marginal = graph.marginals()[self.confounder]
	# 	params.append(marginal)
	#
	# 	# params.append(graph.marginals(**{self.treatment: 0, self.confounder: 0})[self.outcome])
	# 	# params.append(graph.marginals(**{self.treatment: 1, self.confounder: 0})[self.outcome])
	# 	# params.append(graph.marginals(**{self.treatment: 0, self.confounder: 1})[self.outcome])
	# 	# params.append(graph.marginals(**{self.treatment: 1, self.confounder: 1})[self.outcome])
	#
	# 	params.extend(graph.Y.params.reshape(-1).tolist())
	#
	# 	return np.array(params)


	def constraints(self, graph):
		lbs, ubs = [], []

		A = np.eye(5)

		lb, ub = graph.marginal_bounds(self.confounder)
		lbs.append(lb)
		ubs.append(ub)

		for t, c in [(0, 0), (1, 0), (0, 1), (1, 1)]:
			lb, ub = graph.marginal_bounds(self.outcome, **{self.treatment: t, self.confounder: c})
			lbs.append(lb)
			ubs.append(ub)

		return opt.LinearConstraint(A, lbs, ubs, keep_feasible=True)


	def estimate(self, x):
		c = x[0]
		y00, y10, y01, y11 = x[1:]
		return c*(y11 - y01) + (1-c)*(y10 - y00)


	def to_targets(self, x):
		vals = x.tolist()
		return {
			self.confounder: vals[0],
			f'{self.outcome}|{self.treatment}=0,{self.confounder}=0': vals[1],
			f'{self.outcome}|{self.treatment}=1,{self.confounder}=0': vals[2],
			f'{self.outcome}|{self.treatment}=0,{self.confounder}=1': vals[3],
			f'{self.outcome}|{self.treatment}=1,{self.confounder}=1': vals[4],
		}



class ManualConstraints(Confounding_ATE):
	def __init__(self, bounds=None):
		self.bounds = bounds


	def constraints(self, graph):
		lbs, ubs = [], []

		A = np.eye(5)

		lb, ub = self.bounds[self.confounder]
		lbs.append(lb)
		ubs.append(ub)

		for t, c in [(0, 0), (1, 0), (0, 1), (1, 1)]:
			lb, ub = self.bounds[f'{self.outcome}|{self.treatment}={t},{self.confounder}={c}']
			lbs.append(lb)
			ubs.append(ub)

		return opt.LinearConstraint(A, lbs, ubs, keep_feasible=True)



def ate(x):
	c = x[0]
	y00, y10, y01, y11 = x[1:]
	return c * (y11 - y01) + (1 - c) * (y10 - y00)



def ate_lim(b, *, mx=False):
	lb, ub = b.reshape(2, -1)

	A = np.eye(len(lb))

	x0 = (ub - lb) / 2 + lb

	bound_constraint = opt.LinearConstraint(A, lb, ub, keep_feasible=True)

	if mx:
		sol = opt.minimize(lambda x: -ate(x), x0, constraints=[bound_constraint])
		best = -sol.fun
	else:
		sol = opt.minimize(ate, x0, constraints=[bound_constraint])
		best = sol.fun

	return best



def worst_case_diff(x, *, gap=0.05,):# space_wt=0):

	b1, b2 = x.reshape(2, -1)

	min_b1 = ate_lim(b1, mx=False)
	max_b2 = ate_lim(b2, mx=True)

	# reg = 0.
	# if space_wt > 0:
	# 	for b in [b1, b2]:
	# 		lb, ub = b.reshape(2, -1)
	# 		reg += space_wt * np.linalg.norm(ub - lb) ** 2

	return (min_b1 - max_b2 - gap) ** 2# + reg



def test_optim_ate():
	# graph_id = 'confounding'
	# graph = create_graph(graph_id)

	bound_keys = ['V1', 'Y|X=0,V1=0', 'Y|X=1,V1=0', 'Y|X=0,V1=1', 'Y|X=1,V1=1']
	bounds = {
		'V1': (0.8, 0.9),
		'Y|X=0,V1=0': (0.1, 0.3),
		'Y|X=1,V1=0': (0.6, 0.9),
		'Y|X=0,V1=1': (0.2, 0.5),
		'Y|X=1,V1=1': (0.6, 0.9),
	}
	N = len(bound_keys)

	b_param = np.array([bounds[k] for k in bound_keys]).T#.reshape(-1)

	lb, ub = b_param.reshape(2, 1, -1)

	_, mid1, mid2, _ = np.linspace(0, 1, 4).reshape(-1, 1) * (ub - lb) + lb

	x0 = np.concatenate([mid1, mid2]*2)

	A = np.zeros((2*N, 4*N))
	A[:N, :N] = -np.eye(N)
	A[:N, N:2*N] = np.eye(N)
	A[N:, -2*N:-N] = -np.eye(N)
	A[N:, -N:] = np.eye(N)

	b_constraint = opt.LinearConstraint(A, 0.2, np.inf)
	lim_constraint = opt.LinearConstraint(np.eye(4*N), 0., 1., keep_feasible=True)

	sol = opt.minimize(worst_case_diff, x0, constraints=[b_constraint, lim_constraint])
	print(sol)

	b1, b2 = sol.x.reshape(2, 2, -1)

	ate1 = ate_lim(b1, mx=False), ate_lim(b1, mx=True)
	ate2 = ate_lim(b2, mx=False), ate_lim(b2, mx=True)

	bounds1 = {k: v for k, v in zip(bound_keys, b1.T.tolist())}
	bounds2 = {k: v for k, v in zip(bound_keys, b2.T.tolist())}

	print(ate1, ate2)

	# result = [0.6037611143627823, 0.7191040317469358, 0.4108959426685874, 0.3367749057333138, 0.5932250465964617, 0.6237611143627824, 0.7391040317469358, 0.43089594266858744, 0.35677490573331383, 0.6132250465964617, 0.5657600718296564, 0.7385770009465376, 0.3914229594905485, 0.36554397842205016, 0.5644559673929129, 0.5857600718296564, 0.7585770009465376, 0.4114229594905485, 0.3855439784220502, 0.584455967392913]
	# result = np.array(result)
	#
	# b1, b2 = result.reshape(2, -1)
	#
	# b1 = {k: v for k, v in zip(bound_keys, b1.T.tolist())}
	# b2 = {k: v for k, v in zip(bound_keys, b2.T.tolist())}
	pass



class RobustGapOptimization:
	def __init__(self, graph, ):
		pass


