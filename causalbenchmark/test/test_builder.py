
import numpy as np

from ..graphs import create_graph, BayesNet, hparam, node

# from ..graphs.builders import RelativeSpawner, RelativeBuilder, RelativeSCMEnsemble

from ..graphs.builders import MechanismCorrelationBuilder, NiceSCMBuilder, DifficultyBuilder
from ._test_util import example_labels


def test_difficulty():
	
	builder = DifficultyBuilder()
	
	graph_id = 'arrowhead'
	
	spec = {'Y': {'V3': 0, 'X': 1, 'V2': 0}}
	params = builder.generate_scm_params(graph_id, spec)
	print(params)

	spec = {'Y': {'V3': 1, 'X': 1, 'V2': 0}}
	params = builder.generate_scm_params(graph_id, spec)
	print(params)
	
	spec = {'Y': {'V3': 0, 'X': -1, 'V2': 0}}
	params = builder.generate_scm_params(graph_id, spec)
	print(params)

	spec = {'Y': {'V3': 0, 'X': 0, 'V2': -1}}
	params = builder.generate_scm_params(graph_id, spec)
	print(params)



def test_relative_spawner():
	builder = MechanismCorrelationBuilder()
	
	specs = list(builder.spawn_specs('example'))
	
	assert len(specs) == 48
	
	
	
def test_builder():
	graph_id = 'example'
	
	builder = MechanismCorrelationBuilder(seed=1)

	for spec in builder.spawn_specs(graph_id):
		scm = builder.generate_scm_example(graph_id, spec)
		
		assert np.all(spec['Z']['X'] * (scm.Z.param[1] - scm.Z.param[0]) >= 0)
		assert np.all(spec['Y']['X'] * (scm.Y.param[1] - scm.Y.param[0]) >= 0)
		assert np.all(spec['Y']['Z'] * (scm.Y.param[:, 1] - scm.Y.param[:, 0]) >= 0)
		


def test_ensemble():
	graph_id = 'example-confounding'

	builder = MechanismCorrelationBuilder(seed=100)

	specs = list(builder.spawn_specs(graph_id))

	spec = builder._rng.choice(specs)

	N = 5

	# ensemble = RelativeSCMEnsemble(graph_id=graph_id, spec=spec, builder=builder, num_samples=N)
	ensemble = builder.generate_ensemble_scm(graph_id=graph_id, spec=spec, size=N)

	assert len(ensemble) == N

	ates = [sample['Y'] for sample in ensemble.ate_samples('X')]
	
	mn = min(ates)
	mx = max(ates)
	
	lb, ub = ensemble.ate_bounds('Y', 'X')

	assert np.isclose(lb, mn)
	assert np.isclose(ub, mx)

	print(ensemble.verbalize_mechanisms(example_labels))



def test_specs_nice():
	graph_id = 'example-confounding'

	builder = NiceSCMBuilder(seed=100)

	count = builder.spec_count(graph_id)

	if count < 1000:
		specs = list(builder.spawn_specs(graph_id))
		assert len(specs) == count


	specs = set(map(str, builder.sample_specs(graph_id, 100)))
	assert len(specs) == 100



def test_specs_mechcorr():
	graph_id = 'example-confounding'

	builder = MechanismCorrelationBuilder(seed=100)

	count = builder.spec_count(graph_id)

	if count < 1000:
		specs = list(builder.spawn_specs(graph_id))

		assert len(specs) == count

	specs = set(map(str, builder.sample_specs(graph_id, 10)))
	assert len(specs) == 10



def test_correlation():
	builder = MechanismCorrelationBuilder()

	graph_id = 'example-confounding'

	specs = list(builder.spawn_specs(graph_id))

	assert len(specs) == 48


	spec = builder._rng.choice(specs)

	for _ in range(10):

		# g = builder.generate_scm_example(graph_id, spec)

		g = create_graph(graph_id)

		print()
		print(g.get_parameters())
		print(g.X.param)
		print(g.Y.param)

		cXY = g.marginals(X=1)['Y'] - g.marginals(X=0)['Y']
		cYX = g.marginals(Y=1)['X'] - g.marginals(Y=0)['X']

		print(f'cXY = {cXY:.3f} vs cYX = {cYX:.3f}')

		# assert np.isclose(cXY, cYX)




