import numpy as np
from omnidata import Guru
from ..graphs import create_graph
from ..graphs.stories.simpson import SimpsonsParadox


def test_det_fork():
	g = create_graph('det-twocauses')
	ctx = g.context()
	ctx.update(g.brute_force())

	assert np.allclose(ctx['X'] | ctx['V2'], ctx['Y'])

	g = create_graph('det-twocauses', conjunction=True)
	ctx = g.context()
	ctx.update(g.brute_force())

	assert np.allclose(ctx['X'] & ctx['V2'], ctx['Y'])



def test_det_confounding():
	g = create_graph('det-triangle')
	ctx = g.context()
	ctx.update(g.brute_force())

	assert np.allclose(ctx['X'], ctx['V2'])
	assert np.allclose(ctx['X'] | ctx['V2'], ctx['Y'])


	g = create_graph('det-triangle', negation=True)
	ctx = g.context()
	ctx.update(g.brute_force())

	assert np.allclose(ctx['X'], 1-ctx['V2'])
	assert np.allclose(ctx['X'] | ctx['V2'], ctx['Y'])


	g = create_graph('det-triangle', conjunction=True)
	ctx = g.context()
	ctx.update(g.brute_force())

	assert np.allclose(ctx['X'], ctx['V2'])
	assert np.allclose(ctx['X'] & ctx['V2'], ctx['Y'])


	g = create_graph('det-triangle', conjunction=True, negation=True)
	ctx = g.context()
	ctx.update(g.brute_force())

	assert np.allclose(ctx['X'], 1-ctx['V2'])
	assert np.allclose(ctx['X'] & ctx['V2'], ctx['Y'])



def test_det_diamond():

	g = create_graph('det-diamond')
	ctx = g.context()
	ctx.update(g.brute_force())

	assert np.allclose(ctx['X'], ctx['V2'])
	assert np.allclose(ctx['X'], ctx['V3'])
	assert np.allclose(ctx['V2'] | ctx['V3'], ctx['Y'])


	g = create_graph('det-diamond', negation=True)
	ctx = g.context()
	ctx.update(g.brute_force())

	assert np.allclose(ctx['X'], ctx['V2'])
	assert np.allclose(ctx['X'], 1-ctx['V3'])
	assert np.allclose(ctx['V2'] | ctx['V3'], ctx['Y'])


	g = create_graph('det-diamond', negation=True, conjunction=True)
	ctx = g.context()
	ctx.update(g.brute_force())

	assert np.allclose(ctx['X'], ctx['V2'])
	assert np.allclose(ctx['X'], 1-ctx['V3'])
	assert np.allclose(ctx['V2'] & ctx['V3'], ctx['Y'])



def test_det_diamondcut():
	g = create_graph('det-diamondcut')
	ctx = g.context()
	ctx.update(g.brute_force())

	assert np.allclose(ctx['V1'], ctx['X'])
	assert np.allclose(ctx['V1'], ctx['V3'])
	assert np.allclose(ctx['X'] | ctx['V3'], ctx['Y'])

	g = create_graph('det-diamondcut', negation=True)
	ctx = g.context()
	ctx.update(g.brute_force())

	assert np.allclose(ctx['V1'], ctx['X'])
	assert np.allclose(ctx['V1'], 1 - ctx['V3'])
	assert np.allclose(ctx['X'] | ctx['V3'], ctx['Y'])

	g = create_graph('det-diamondcut', negation=True, conjunction=True)
	ctx = g.context()
	ctx.update(g.brute_force())

	assert np.allclose(ctx['V1'], ctx['X'])
	assert np.allclose(ctx['V1'], 1 - ctx['V3'])
	assert np.allclose(ctx['X'] & ctx['V3'], ctx['Y'])


def test_arrowhead():

	g = create_graph('arrowhead')
	print()
	print(g)

	vs = list(g.variables())

	assert len(vs) == 4

	# print(vs)







def test_simpson():

	for _ in range(10):

		# g = create_graph('simpson')#, overlap=0.5, width=0.5, offset=0.2, overlap_offset=0.4)
		g = SimpsonsParadox()#, overlap=0.5, width=0.5, offset=0.2, overlap_offset=0.4)

		stats = g.simpsons_paradox_stats()

		[[worst_control, best_control, agg_control],
		 [worst_treated, best_treated, agg_treated]] = stats

		assert agg_treated < agg_control
		assert worst_treated > worst_control
		assert best_treated > best_control







# def test_simpson_natural():
# 	for i in range(10):
#
#
# 		g = create_graph('simpson', overlap=0.5, width=0.5, offset=0.2, overlap_offset=0.4, prior=0.7, gap=0.3, aggregate_offset=0.1)
#
# 		true_control, true_treated = g.marginals(X=0)['Y'], g.marginals(X=1)['Y']
#
# 		pred_control, pred_treated = g._expected_aggregate(g.Y.p)
#
# 		print()
# 		print(true_control, true_treated)
# 		print(pred_control, pred_treated)
# 		assert np.isclose(true_control, pred_control)
# 		assert np.isclose(true_treated, pred_treated)
	
		
	


# def test_confounding():
# 	simpson_count = 0
# 	for i in range(10):
# 		context = Guru(create_graph('confounding'), size=10000)
#
# 		X = context['X']
# 		Z = context['Z']
# 		Y = context['Y']
#
# 		dat = np.column_stack((X, Z, Y))
# 		context.clear()
#
# 		def check_simpsons_paradox_numpy(data):
# 			corr_XY = np.corrcoef(data[:, 0], data[:, 2])[0, 1]
# 			print(corr_XY)
# 			Z1_indices = data[:, 1] > 0
# 			Z0_indices = data[:, 1] <= 0
#
#
# 			corr_XY_given_Z1 = np.corrcoef(data[Z1_indices, 0], data[Z1_indices, 2])[0, 1]
# 			corr_XY_given_Z0 = np.corrcoef(data[Z0_indices, 0], data[Z0_indices, 2])[0, 1]
# 			print(corr_XY_given_Z1,"    ",corr_XY_given_Z0)
#
# 			return (corr_XY * corr_XY_given_Z1 < 0 and corr_XY * corr_XY_given_Z0 < 0)
#
# 		if check_simpsons_paradox_numpy(dat):
# 			simpson_count += 1
#
#
# 	assert simpson_count > 9






# def test_collision():
# 	def ground_truth_collision(beta_yx, beta_yz):
# 		return 0
#
# 	beta_list = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99]
# 	for yx in beta_list:
# 		for yz in beta_list:
# 			context = Guru(create_graph('collision', beta_YX = yz, beta_YZ = yz), size = 1)
# 			# context['Y']
# 			assert ground_truth_collision(yx, yz) == 0




	# assert context['Y']
	#
	# context = Guru(Diamond(), size=10)
	#
	#
	# assert np.all(context['Y'] == context['X'])


# def test_dowhy():
# 	pass


# def test_simpsons():
# 	for _ in range(10):
#
# 		g = create_graph('simpson', )
#
# 		stats = g.simpsons_paradox(1, 1)
#
# 		[[worst_control, best_control, agg_control],
# 		 [worst_treated, best_treated, agg_treated]] = stats
#
# 		assert agg_control == best_control and worst_treated == agg_treated
#
# 		stats = g.simpsons_paradox(1, -1)
#
# 		[[worst_control, best_control, agg_control],
# 		 [worst_treated, best_treated, agg_treated]] = stats
#
# 		assert agg_control == agg_treated and worst_treated == agg_treated
#
# 		severity, gap = np.random.rand(), np.random.rand()
# 		# bias_t, bias_c = 0.05, 0.3
# 		stats = g.simpsons_paradox(severity, gap)
#
# 		[[worst_control, best_control, agg_control],
# 		 [worst_treated, best_treated, agg_treated]] = stats
#
# 		assert agg_control >= agg_treated and worst_treated >= worst_control and best_treated >= best_control
#
# 		stats = g.simpsons_paradox(severity, -gap)
#
# 		[[worst_control, best_control, agg_control],
# 		 [worst_treated, best_treated, agg_treated]] = stats
#
# 		assert agg_control < agg_treated and worst_treated >= worst_control and best_treated >= best_control



