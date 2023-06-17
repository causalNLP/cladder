from ..graphs import create_graph
from ..queries import create_query
import numpy as np
import time

from ..graphs.builders import MechanismCorrelationBuilder

from . import _test_util



def test_constrained():

	graph_id = 'example-confounding'

	builder = MechanismCorrelationBuilder(seed=11)

	specs = list(builder.spawn_specs(graph_id))
	specs = np.random.choice(specs, size=5)
	specs = [{'Z': 1, 'X': {'Z': 1}, 'Y': {'X': 0, 'Z': -1}}]

	ana_time = 0.
	mc_time = 0.

	ana = []
	mc = []

	size = 10

	print()
	for spec in specs:
		tick = time.time()
		scm_ana = builder.generate_constrained_scm(graph_id, spec)#, method='slsqp')
		ana.append(scm_ana.ate_bounds('Y', 'X'))
		ana_time += time.time() - tick

		tick = time.time()
		scm_mc = builder.generate_ensemble_scm(graph_id, spec, size=size)
		mc.append(scm_mc.ate_bounds('Y', 'X'))
		mc_time += time.time() - tick

	print(f'Analytical: {ana_time:.2f}')
	print(f'Monte Carlo: {mc_time:.2f} (would be about {mc_time * 100/size:.2f} for size 100)')
	for spec, a, m in zip(specs, ana, mc):
		print(spec)
		print(f'{a[0]:.2f} <= {m[0]:.2f} <= {m[1]:.2f} <= {a[1]:.2f}')
		assert a[0] <= m[0] <= m[1] <= a[1]


































####################################################################################################





# from omnidata import Guru
# from causalbenchmark.queries.solvers import ace

# def test_ace_solver():
#     context = Guru(create_graph('confounding'), size=100000)
#     X = context['X']
#     Z = context['Z']
#     Y = context['Y']
#     dat = np.column_stack((X, Z, Y))
#
#     def check_simpsons_paradox_numpy(data):
#         corr_XY = np.corrcoef(data[:, 0], data[:, 2])[0, 1]
#         print(corr_XY)
#         Z1_indices = data[:, 1] > 0
#         Z0_indices = data[:, 1] <= 0
#
#         corr_XY_given_Z1 = np.corrcoef(data[Z1_indices, 0], data[Z1_indices, 2])[0, 1]
#         corr_XY_given_Z0 = np.corrcoef(data[Z0_indices, 0], data[Z0_indices, 2])[0, 1]
#
#         return (corr_XY * corr_XY_given_Z1 < 0 and corr_XY * corr_XY_given_Z0 < 0)
#
#     if check_simpsons_paradox_numpy(dat):
#         ace_est = ace(context, 'X', 'Y')
#         print(ace_est)
#         assert (-0.6 < ace_est < -0.4)

# def test_ace_solver():
#     context = Guru(create_graph('confounding'), size=100000)
#     X = context['X']
#     Z = context['Z']
#     Y = context['Y']
#     dat = np.column_stack((X, Z, Y))
#
#     def check_simpsons_paradox_numpy(data):
#         corr_XY = np.corrcoef(data[:, 0], data[:, 2])[0, 1]
#         print(corr_XY)
#         Z1_indices = data[:, 1] > 0
#         Z0_indices = data[:, 1] <= 0
#
#         corr_XY_given_Z1 = np.corrcoef(data[Z1_indices, 0], data[Z1_indices, 2])[0, 1]
#         corr_XY_given_Z0 = np.corrcoef(data[Z0_indices, 0], data[Z0_indices, 2])[0, 1]
#
#         return (corr_XY * corr_XY_given_Z1 < 0 and corr_XY * corr_XY_given_Z0 < 0)
#
#     if check_simpsons_paradox_numpy(dat):
#         ace_est = ace(context, 'X', 'Y')
#         print(ace_est)
#         assert (-0.6 < ace_est < -0.4)









