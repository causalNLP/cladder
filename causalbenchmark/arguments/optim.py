
import numpy as np
from scipy import optimize as opt

from .systems import *


class RobustGapOptimization:
	def __init__(self, system, *, #bounds1=None, bounds2=None, fixed1=None, fixed2=None,
	             gap=0.05, width=0.1, seed=1, eps=1e-4):
		self.gen = np.random.RandomState(seed)
		self.gap = gap
		self.width = width
		self.eps = eps
		self.system = system
		# self.bounds1 = bounds1
		# self.bounds2 = bounds2
		# self.fixed1 = fixed1
		# self.fixed2 = fixed2


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

		bound_constraint = opt.LinearConstraint(A, lb, ub, )#keep_feasible=True)

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
		param = np.asarray([lims.get(k, [0., 1.]) for k in dofs]).T
		eps = np.asarray([-eps, eps]).reshape(2, 1)
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
		lim_constraint = opt.LinearConstraint(np.eye(2*(N1+N2)), 0., 1., )#keep_feasible=True)

		return [b_constraint, lim_constraint]


	def _toplevel_package(self, sol):
		b1, b2 = sol.x[:self.N1], sol.x[self.N1:]

		bounds1 = dict(zip([k for k in self._term1_dofs if self.fixed1 is None or k not in self.fixed1],
		                   b1.reshape(2, -1).T.tolist()))
		if self.fixed1 is not None:
			bounds1.update({k: self.bounds1.get(k, [0., 1.]) for k in self.fixed1})
		bounds2 = dict(zip([k for k in self._term2_dofs if self.fixed2 is None or k not in self.fixed2],
		                   b2.reshape(2, -1).T.tolist()))
		if self.fixed2 is not None:
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



def test_optim_gap():
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
	
	# system = IVSystem()
	# gen = np.random.RandomState(11)
	# suggested = {k: sorted(gen.uniform(0., 1., size=2).tolist()) for k in system.ate_dofs}

	optim = ATEGapOptimization(system, bounds1=suggested.copy(), bounds2=suggested.copy(),
	                           fixed1=fixed1, fixed2=fixed2, gap=0.05, width=0.1)

	bounds1, bounds2 = optim.solve()

	print(bounds1)
	print(bounds2)

	ate1 = optim.estimand_bounds1(bounds1)
	ate2 = optim.estimand_bounds2(bounds2)

	print(ate1)
	print(ate2)



class PathGapOptimization(RobustGapOptimization):
	def __init__(self, *args, direct_is_higher=False, bounds=None, fixed=None,
	             space=0.1, space_wt=1., **kwargs):
		if bounds is None:
			bounds = {}
		super().__init__(*args, **kwargs)
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
		lim_constraint = opt.LinearConstraint(np.eye(2*N), 0., 1.,)# keep_feasible=True)

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
		return self.system.nde_dofs if self.direct_is_higher else self.system.nie_fast


	@property
	def _term2_dofs(self):
		return self.system.nie_dofs if self.direct_is_higher else self.system.nde_fast


	@property
	def _term1_estimand(self):
		return self.system.nde_fast if self.direct_is_higher else self.system.nie_fast


	@property
	def _term2_estimand(self):
		return self.system.nie_fast if self.direct_is_higher else self.system.nde_fast



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


