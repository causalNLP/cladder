
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



class RobustGapOptimization:
	def __init__(self, system, *, bounds1=None, bounds2=None, fixed1=None, fixed2=None,
	             gap=0.05, width=0.1, eps=1e-4):
		self.gap = gap
		self.width = width
		self.eps = eps
		self.system = system
		self.bounds1 = bounds1
		self.bounds2 = bounds2
		self.fixed1 = fixed1
		self.fixed2 = fixed2


	def _toplevel_initial(self):
		raise NotImplementedError


	def _toplevel_constraints(self):
		raise NotImplementedError


	def _toplevel_objective(self, x):
		raise NotImplementedError


	@property
	def _term1_estimand(self):
		raise NotImplementedError

	@property
	def _term2_estimand(self):
		raise NotImplementedError

	@property
	def _term1_dofs(self):
		raise NotImplementedError

	@property
	def _term2_dofs(self):
		raise NotImplementedError


	@staticmethod
	def estimand_limit(estimand, b, *, upper=False):
		lb, ub = b.reshape(2, -1)

		A = np.eye(len(lb))

		x0 = (ub - lb) / 2 + lb

		bound_constraint = opt.LinearConstraint(A, lb, ub, keep_feasible=True)

		if upper:
			sol = opt.minimize(lambda x: -estimand(x), x0, constraints=[bound_constraint])
			best = -sol.fun
		else:
			sol = opt.minimize(estimand, x0, constraints=[bound_constraint])
			best = sol.fun

		return best


	def solve(self):
		x0 = self._toplevel_initial()
		constraints = self._toplevel_constraints()

		sol = opt.minimize(self._toplevel_objective, x0, constraints=constraints)

		if not sol.success:
			print('WARNING: optimization failed')
			print(sol)

		return self._toplevel_package(sol)


	def estimand_bounds1(self, bounds, eps=0.):
		b = self.process_bounds1(bounds, eps=eps)
		limits = self.estimand_limit(self._term1_estimand, b, upper=False), \
			self.estimand_limit(self._term1_estimand, b, upper=True)
		return limits


	def estimand_bounds2(self, bounds, eps=0.):
		b = self.process_bounds2(bounds, eps=eps)
		limits = self.estimand_limit(self._term2_estimand, b, upper=False), \
			self.estimand_limit(self._term2_estimand, b, upper=True)
		return limits


	@staticmethod
	def process_bounds(lims, dofs, *, eps=1e-4):
		param = np.array([lims.get(k, [0., 1.]) for k in dofs]).T
		eps = np.array([-eps, eps]).reshape(2, 1)
		param = param + eps
		param = np.clip(param, 0., 1.)
		return param # 2 x n


	def process_bounds1(self, bounds, eps=0.):
		return self.process_bounds(bounds, self._term1_dofs, eps=eps)


	def process_bounds2(self, bounds, eps=0.):
		return self.process_bounds(bounds, self._term2_dofs, eps=eps)


	def _prepare_initial(self, param, dofs, fixed=None):
		lb, ub = param.reshape(2, 1, -1)
		lim_l, mid_l, mid_u, lim_u = np.linspace(0, 1, 4).reshape(-1, 1) * (ub - lb) + lb

		l, u = mid_l, mid_u

		sel = None
		if fixed is not None and any(d in fixed for d in dofs):
			sel = np.asarray([k not in fixed for k in dofs]).astype(bool)
			l[~sel] = lim_l[~sel]
			u[~sel] = lim_u[~sel]
			sel = np.concatenate([sel, sel])

		param = np.concatenate([l, u])
		return param, sel



class ATEGapOptimization(RobustGapOptimization):
	def __init__(self, *args, bounds1=None, bounds2=None, fixed1=None, fixed2=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.bounds1 = bounds1
		self.bounds2 = bounds2
		self.fixed1 = fixed1
		self.fixed2 = fixed2


	def _toplevel_initial(self):
		self.params1, self.sel1 = self._prepare_initial(self.process_bounds1(self.bounds1, eps=self.eps),
		                                                self._term1_dofs, self.fixed1)
		self.N1 = len(self.params1) if self.sel1 is None else self.sel1.sum()

		self.params2, self.sel2 = self._prepare_initial(self.process_bounds2(self.bounds2, eps=self.eps),
		                                                self._term2_dofs, self.fixed2)
		self.N2 = len(self.params2) if self.sel2 is None else self.sel2.sum()

		return np.concatenate([self.params1 if self.sel1 is None else self.params1[self.sel1],
		                       self.params2 if self.sel2 is None else self.params2[self.sel2]], axis=0)


	def _toplevel_constraints(self):
		N1, N2 = self.N1//2, self.N2//2
		A = np.zeros((N1+N2, 2*(N1+N2)))
		A[:N1, :N1] = -np.eye(N1)
		A[:N1, N1:2 * N1] = np.eye(N1)
		A[N1:, -2 * N2:-N2] = -np.eye(N2)
		A[N1:, -N2:] = np.eye(N2)

		b_constraint = opt.LinearConstraint(A, self.width, np.inf)
		lim_constraint = opt.LinearConstraint(np.eye(2*(N1+N2)), 0., 1., keep_feasible=True)

		return [b_constraint, lim_constraint]


	def _toplevel_package(self, sol):
		b1, b2 = sol.x[:self.N1], sol.x[self.N1:]

		bounds1 = dict(zip([k for k in self._term1_dofs if self.fixed1 is None or k not in self.fixed1],
		                   b1.reshape(2, -1).T.tolist()))
		bounds1.update({k: self.bounds1.get(k, [0., 1.]) for k in self.fixed1})
		bounds2 = dict(zip([k for k in self._term2_dofs if self.fixed2 is None or k not in self.fixed2],
		                   b2.reshape(2, -1).T.tolist()))
		bounds2.update({k: self.bounds2.get(k, [0., 1.]) for k in self.fixed2})
		return bounds1, bounds2


	def _toplevel_objective(self, x):
		b1, b2 = x[:self.N1], x[self.N1:]

		if self.sel1 is not None:
			self.params1[self.sel1] = b1
			b1 = self.params1
		if self.sel2 is not None:
			self.params2[self.sel2] = b2
			b2 = self.params2

		min_b1 = self.estimand_limit(self._term1_estimand, b1, upper=False)
		max_b2 = self.estimand_limit(self._term2_estimand, b2, upper=True)

		return (min_b1 - max_b2 - self.gap) ** 2


	@property
	def _term1_dofs(self):
		return self.system.ate_dofs


	@property
	def _term2_dofs(self):
		return self.system.ate_dofs


	@property
	def _term1_estimand(self):
		return self.system.ate_fast


	@property
	def _term2_estimand(self):
		return self.system.ate_fast



class PathGapOptimization(RobustGapOptimization):
	def __init__(self, *args, direct_is_higher=False, bounds=None, fixed=None, space=0.1, space_wt=1., seed=1, **kwargs):
		if bounds is None:
			bounds = {}
		super().__init__(*args, **kwargs)
		self.gen = np.random.RandomState(seed)
		self.direct_is_higher = direct_is_higher
		self.bounds = bounds
		self.fixed = fixed
		self.space = space
		self.space_wt = space_wt


	def _toplevel_initial(self):
		keys = list(set(self._term1_dofs) | set(self._term2_dofs))

		# the random sampling is important to avoid getting stuck in a saddle point or unstable equilibrium
		params = np.asarray([self.bounds.get(k, [self.gen.uniform(0.,0.01), 1.-self.gen.uniform(0.,0.01)])
		                     for k in keys]).T.reshape(-1)

		self.param_sel1 = np.asarray([keys.index(k) for k in self._term1_dofs])
		self.param_sel1 = np.concatenate([self.param_sel1, self.param_sel1 + len(keys)])
		self.params1, self.fixed_sel1 = self._prepare_initial(params[self.param_sel1], keys, self.fixed)

		self.param_sel2 = np.asarray([keys.index(k) for k in self._term2_dofs])
		self.param_sel2 = np.concatenate([self.param_sel2, self.param_sel2 + len(keys)])
		self.params2, self.fixed_sel2 = self._prepare_initial(params[self.param_sel2], keys, self.fixed)

		params[self.param_sel1] = self.params1
		params[self.param_sel2] = self.params2

		self.keys = keys
		return params


	def _toplevel_constraints(self):
		N = len(self.keys)
		A = np.zeros((N, 2*N))
		A[:N, :N] = -np.eye(N)
		A[:N, -N:] = np.eye(N)

		b_constraint = opt.LinearConstraint(A, self.width, np.inf)
		lim_constraint = opt.LinearConstraint(np.eye(2*N), 0., 1., keep_feasible=True)

		return [b_constraint, lim_constraint]



	def _toplevel_objective(self, x):
		b1, b2 = x[self.param_sel1], x[self.param_sel2]

		if self.fixed_sel1 is not None:
			self.params1[self.fixed_sel1] = b1
			b1 = self.params1
		if self.fixed_sel2 is not None:
			self.params2[self.fixed_sel2] = b2
			b2 = self.params2

		min_b1 = self.estimand_limit(self._term1_estimand, b1, upper=False)
		max_b2 = self.estimand_limit(self._term2_estimand, b2, upper=True)

		loss = (min_b1 - max_b2 - self.gap) ** 2

		if self.space_wt > 0.:
			max_b1 = self.estimand_limit(self._term1_estimand, b1, upper=True)
			min_b2 = self.estimand_limit(self._term2_estimand, b2, upper=False)
			# space = (max_b1 - min_b1) / (max_b2 + min_b2 + 1e-8)
			loss += self.space_wt * ((max_b1 - min_b1 - self.space) ** 2 + (max_b2 - min_b2 - self.space) ** 2)

		return loss


	def _toplevel_package(self, sol):
		b1, b2 = sol.x[self.param_sel1], sol.x[self.param_sel2]

		bounds1 = dict(zip([k for k in self._term1_dofs if self.fixed is None or k not in self.fixed],
		                   b1.reshape(2, -1).T.tolist()))
		if self.fixed is not None:
			bounds1.update({k: self.bounds.get(k, [0., 1.]) for k in self.fixed if k in self._term1_dofs})
		bounds2 = dict(zip([k for k in self._term2_dofs if self.fixed is None or k not in self.fixed],
		                   b2.reshape(2, -1).T.tolist()))
		if self.fixed is not None:
			bounds2.update({k: self.bounds.get(k, [0., 1.]) for k in self.fixed if k in self._term2_dofs})
		return bounds1, bounds2


	@property
	def _term1_dofs(self):
		return self.system.nie_dofs


	@property
	def _term2_dofs(self):
		return self.system.nde_dofs


	@property
	def _term1_estimand(self):
		return self.system.nde_fast if self.direct_is_higher else self.system.nie_fast


	@property
	def _term2_estimand(self):
		return self.system.nie_fast if self.direct_is_higher else self.system.nde_fast



class ConfoundingSystem:
	ate_dofs = r'\sum_{V1=v} P(V1=v)*[P(Y=1|V1=v,X=1) - P(Y=1|V1=v, X=0)]'
	ate_dofs = ['V1', 'Y|X=0,V1=0', 'Y|X=1,V1=0', 'Y|X=0,V1=1', 'Y|X=1,V1=1']

	@staticmethod
	def ate_fast(x):
		c = x[0]
		y00, y10, y01, y11 = x[1:]
		return c * (y11 - y01) + (1 - c) * (y10 - y00)



class IVSystem:
	ate_str = '[P(Y=1|V2=1)-P(Y=1|V2=0)]/[P(X=1|V2=1)-P(X=1|V2=0)]'
	ate_dofs = ['Y|V2=1', 'Y|V2=0', 'X|V2=1', 'X|V2=0']

	@staticmethod
	def ate_fast(x):
		y1, y0, x1, x0 = x
		return (y1 - y0) / (x1 - x0)



class FrontdoorSystem:
	ate_str = r'\sum_{V3 = v} [P(V3 = v|X = 1) - P(V3 = v|X = 0)] * [\sum_{X = h} P(Y = 1|X = h,V3 = v)*P(X = h)]'
	ate_dofs = ['X', 'V3|X=1', 'V3|X=0', 'Y|X=0,V3=0', 'Y|X=1,V3=0', 'Y|X=0,V3=1', 'Y|X=1,V3=1']

	@staticmethod
	def ate_fast(x):
		x, v31, v30, y00, y10, y01, y11 = x
		return (v31 - v30) * (x * (y11 - y10) + (1 - x) * (y01 - y00))



class MediationSystem:
	ate_str = 'P(Y=1|X=1) - P(Y=1|X=0)'
	ate_dofs = ['Y|X=0', 'Y|X=1']

	@staticmethod
	def ate_fast(x):
		y0, y1 = x
		return y1 - y0


	nde_str = '\sum_{V2=v} P(V2=v|X=0)*[P(Y=1|X=1,V2=v) - P(Y=1|X=0, V2=v)]'
	nde_dofs = ['V2|X=0', 'Y|X=0,V2=0', 'Y|X=1,V2=0', 'Y|X=0,V2=1', 'Y|X=1,V2=1']

	@staticmethod
	def nde_fast(x):
		v20, y00, y10, y01, y11 = x
		return v20 * (y11 - y01) + (1 - v20) * (y10 - y00)


	nie_str = '\sum_{V2 = v} P(Y=1|X =0,V2 = v)*[P(V2 = v | X = 1) − P(V2 = v | X = 0)]'
	nie_dofs = ['Y|X=0,V2=0', 'Y|X=0,V2=1', 'V2|X=1', 'V2|X=0']

	@staticmethod
	def nie_fast(x):
		y00, y01, v21, v20 = x
		return (y01 - y00) * (v21 - v20)



def test_optim_gap():
	# graph_id = 'confounding'
	# graph = create_graph(graph_id)

	bound_keys = ['V1', 'Y|X=0,V1=0', 'Y|X=1,V1=0', 'Y|X=0,V1=1', 'Y|X=1,V1=1']
	suggested = {
		'V1': [0.8, 0.9],
		'Y|X=0,V1=0': [0.1, 0.3],
		'Y|X=1,V1=0': [0.6, 0.9],
		'Y|X=0,V1=1': [0.2, 0.5],
		'Y|X=1,V1=1': [0.6, 0.9],
	}

	fixed1 = None
	fixed1 = ['Y|X=0,V1=0']
	# fixed1 = {'Y|X=0,V1=0': [0.4, 0.4]}

	fixed2 = None
	fixed2 = ['V1', 'Y|X=0,V1=1']

	system = ConfoundingSystem()

	optim = RobustGapOptimization(system, bounds1=suggested.copy(), bounds2=suggested.copy(),
	                              fixed1=fixed1, fixed2=fixed2, gap=0.05, width=0.1)

	bounds1, bounds2 = optim.solve()

	print(bounds1)
	print(bounds2)

	ate1 = optim.estimand_bounds1(bounds1)
	ate2 = optim.estimand_bounds2(bounds2)

	print(ate1)
	print(ate2)



def test_optim_paths():

	system = MediationSystem()

	optim = PathGapOptimization(system, gap=0.05, width=0.1)

	bounds1, bounds2 = optim.solve()

	print(bounds1)
	print(bounds2)

	lim1 = optim.estimand_bounds1(bounds1)
	lim2 = optim.estimand_bounds2(bounds2)

	print(lim1)
	print(lim2)




# class OldRobustGapOptimization(RobustGapOptimization): # ATE
# 	dofs1 = ['V1', 'Y|X=0,V1=0', 'Y|X=1,V1=0', 'Y|X=0,V1=1', 'Y|X=1,V1=1']
# 	dofs2 = dofs1
#
# 	@classmethod
# 	def bounds_from_dict(cls, lims, *, eps=1e-4):
# 		b_param = np.array([lims.get(k, [0., 1.]) for k in cls.dofs]).T
# 		lb, ub = b_param.reshape(2, 1, -1)
# 		lb, ub = lb-eps, ub+eps
# 		lb, ub = np.clip(lb, 0., 1.), np.clip(ub, 0., 1.)
# 		return lb, ub
#
#
# 	def _initial_bounds(self):
# 		bounds1 = self.suggested.copy()
# 		bounds1.update(self.fixed1)
# 		lb1, ub1 = self.bounds_from_dict(bounds1, eps=self.eps)
# 		lim_l1, mid_l1, mid_u1, lim_u1 = np.linspace(0, 1, 4).reshape(-1, 1) * (ub1 - lb1) + lb1
#
# 		l1, u1 = mid_l1, mid_u1
#
# 		bounds2 = self.suggested.copy()
# 		bounds2.update(self.fixed2)
# 		lb2, ub2 = self.bounds_from_dict(bounds2, eps=self.eps)
# 		lim_l2, mid_l2, mid_u2, lim_u2 = np.linspace(0, 1, 4).reshape(-1, 1) * (ub2 - lb2) + lb2
#
# 		l2, u2 = mid_l2, mid_u2
#
# 		sel1 = None
# 		if len(self.fixed1):
# 			sel1 = np.asarray([k not in self.fixed1 for k in self.dofs]).astype(bool)
# 			l1[~sel1] = lim_l1[~sel1]
# 			u1[~sel1] = lim_u1[~sel1]
# 			sel1 = np.concatenate([sel1, sel1])
# 		self.sel1 = sel1
#
# 		sel2 = None
# 		if len(self.fixed2):
# 			sel2 = np.asarray([k not in self.fixed2 for k in self.dofs]).astype(bool)
# 			l2[~sel2] = lim_l2[~sel2]
# 			u2[~sel2] = lim_u2[~sel2]
# 			sel2 = np.concatenate([sel2, sel2])
# 		self.sel2 = sel2
#
# 		self.param1 = np.concatenate([l1, u1])
# 		self.param2 = np.concatenate([l2, u2])
#
# 		self.N1 = len(self.param1) if self.sel1 is None else self.sel1.sum()
# 		self.N2 = len(self.param2) if self.sel2 is None else self.sel2.sum()
#
# 		return np.concatenate([self.param1 if self.sel1 is None else self.param1[self.sel1],
# 		                       self.param2 if self.sel2 is None else self.param2[self.sel2]], axis=0)
#
#
# 	def _toplevel_constraints(self):
# 		N1, N2 = self.N1//2, self.N2//2
# 		A = np.zeros((N1+N2, 2*(N1+N2)))
# 		A[:N1, :N1] = -np.eye(N1)
# 		A[:N1, N1:2 * N1] = np.eye(N1)
# 		A[N1:, -2 * N2:-N2] = -np.eye(N2)
# 		A[N1:, -N2:] = np.eye(N2)
#
# 		b_constraint = opt.LinearConstraint(A, self.width, np.inf)
# 		lim_constraint = opt.LinearConstraint(np.eye(2*(N1+N2)), 0., 1., keep_feasible=True)
#
# 		return [b_constraint, lim_constraint]
#
#
# 	def compute_estimand(self, x):
# 		c = x[0]
# 		y00, y10, y01, y11 = x[1:]
# 		return c*(y11 - y01) + (1-c)*(y10 - y00)
#
#
# 	@staticmethod
# 	def estimand_limit(estimand, b, *, upper=False):
# 		lb, ub = b.reshape(2, -1)
#
# 		A = np.eye(len(lb))
#
# 		x0 = (ub - lb) / 2 + lb
#
# 		bound_constraint = opt.LinearConstraint(A, lb, ub, keep_feasible=True)
#
# 		if upper:
# 			sol = opt.minimize(lambda x: -estimand(x), x0, constraints=[bound_constraint])
# 			best = -sol.fun
# 		else:
# 			sol = opt.minimize(estimand, x0, constraints=[bound_constraint])
# 			best = sol.fun
#
# 		return best
#
#
# 	def toplevel_objective(self, x):
# 		b1, b2 = x[:self.N1], x[self.N1:]
#
# 		if self.sel1 is not None:
# 			self.param1[self.sel1] = b1
# 			b1 = self.param1
# 		if self.sel2 is not None:
# 			self.param2[self.sel2] = b2
# 			b2 = self.param2
#
# 		min_b1 = self.estimand_limit(self.compute_estimand, b1, upper=False)
# 		max_b2 = self.estimand_limit(self.compute_estimand, b2, upper=True)
#
# 		return (min_b1 - max_b2 - self.gap) ** 2  # + reg
#
#
# 	def solve(self):
# 		x0 = self._initial_bounds()
# 		constraints = self._toplevel_constraints()
#
# 		sol = opt.minimize(self.toplevel_objective, x0, constraints=constraints)
#
# 		if not sol.success:
# 			print('WARNING: optimization failed')
# 			print(sol)
#
# 		b1, b2 = sol.x[:self.N1], sol.x[self.N1:]
#
# 		bounds1 = dict(zip([k for k in self.dofs if k not in self.fixed1],
# 		                   b1.reshape(2, -1).T.tolist()))
# 		bounds1.update(self.fixed1)
# 		bounds2 = dict(zip([k for k in self.dofs if k not in self.fixed2],
# 		                   b2.reshape(2, -1).T.tolist()))
# 		bounds2.update(self.fixed2)
#
# 		return bounds1, bounds2
#
#
# 	def estimand_bounds(self, bounds):
# 		b = np.concatenate(self.bounds_from_dict(bounds))
# 		limits = self.estimand_limit(self.compute_estimand, b, upper=False), \
# 			self.estimand_limit(self.compute_estimand, b, upper=True)
# 		return limits
#
#
#
