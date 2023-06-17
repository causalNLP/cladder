import numpy as np

from ..graphs import create_graph, BayesNet, hparam, node, register_graph


@register_graph('example')
class Example(BayesNet):
	description_template = hparam('{Zname} mediates the effect of {Xname} on {Yname}.')
	X = node()
	Z = node(X)
	Y = node(X, Z)



def test_something():
	g = create_graph('confounding')
	print(g)



def test_example():

	g = create_graph('example')

	print()
	print(g) # p(X) p(Z | X) p(Y | X, Z)
	print(g.X) # p(X)
	print(g.Z) # p(Z | X)

	for s in g.signatures():
		print(s)

	for n in g.variables():
		print(n)

	# p(X)
	# p(Z | X)
	# p(Y | X, Z)

	print(g.X.param)
	g.X.param = 0.5
	print(g.X.param)
	assert g.X.param == 0.5

	print(g.sample(1000)['X'].mean())

	assert g.sample(1000)['X'].mean() < 0.7 # should be close to 0.5



def test_example_mech():

	g = create_graph('example')

	print()
	print(g) # p(X) p(Z | X) p(Y | X, Z)
	print(g.X) # p(X)
	print(g.Z) # p(Z | X)

	assert np.isclose(g.Z.prob(X=0), g.Z.param[0])
	assert np.isclose(g.Z.prob(0, X=0), 1 - g.Z.param[0])
	assert np.isclose(g.Z.prob(X=1), g.Z.param[1])
	assert np.isclose(g.Z.prob(0, X=1), 1 - g.Z.param[1])
	assert np.isclose(g.Y.prob(X=1, Z=1), g.Y.param[1,1])
	assert np.isclose(g.Y.prob(X=0, Z=1), g.Y.param[0,1])
	assert np.isclose(g.Y.prob(X=1, Z=0), g.Y.param[1,0])
	assert np.isclose(g.Y.prob(X=0, Z=0), g.Y.param[0,0])



def test_intervention():

	g = create_graph('example', params={'X': 0.5, 'Z': [0,1]})
	gi = g.intervene(Z=1) # intervene changing Z to 1

	print()
	print(g)
	print(gi)

	print(g.sample(1000)['Z'].mean())
	print(gi.sample(1000)['Z'].mean())

	assert g.sample(1000)['Z'].mean() < 0.7 # should be close to 0.5
	assert gi.sample(1000)['Z'].mean() == 1. # should be close to 1.0



def test_conditional():

	g = create_graph('example', params={'X': 0.5, 'Z': [0,1]})
	g.Y.param = [0, 1, 0, 1]

	m = g.marginals()

	print()
	print(m)
	assert m['X'] == 0.5 and m['Y'] == 0.5 and m['Z'] == 0.5

	gi = g.intervene(Z=1) # intervene changing Z to 1

	mi = gi.marginals()
	print(mi)
	assert mi['X'] == 0.5 and mi['Y'] == 1 and mi['Z'] == 1



def test_marginals():

	g = create_graph('example')

	marginals = g.marginals()

	samples = g.sample(10000)
	estimated = {name: samples[name].mean() for name in marginals}

	print()
	print(marginals)
	print(estimated)

	for name in g.variable_names():
		assert np.isclose(marginals[name], estimated[name], atol=0.1)



def test_ate():

	g = create_graph('example')

	ate = g.ate('X')

	assert np.isclose(ate['Z'], g.Z.param[1] - g.Z.param[0])

	do_1 = g.intervene(X=1).sample(10000)
	do_0 = g.intervene(X=0).sample(10000)
	estimated = {name: do_1[name].mean() - do_0[name].mean() for name in ate}

	print()
	print(ate)
	print(estimated)

	for name in g.variable_names():
		assert np.isclose(ate[name], estimated[name], atol=0.1)




























