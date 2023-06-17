import json
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from collections import Counter
from omnibelt import export, load_export, save_json, load_json
import omnifig as fig
import numpy as np
import pandas as pd

from . import util
from .graphs import create_graph
from .graphs.builders import NiceSCMBuilder, RandomBuilder
from .verbal import validate_story, load_story
# from .graphs.builders import RelativeSpawner, RelativeBuilder, RelativeSCMEnsemble
from .queries import create_query, QueryFailedError



@fig.script('summary')
def dataset_summary(config, data_path=None, data=None):
	'''
	Generate (and report) a summary of the dataset.
	:param config:
	:return:
	'''

	if data is None:
		if data_path is None:
			data_path = config.pull('path')
		data_path = Path(data_path).expanduser()

		if not data_path.exists():
			raise FileNotFoundError(data_path)

		print(f'Loading data file {data_path}...', end='')

		try:
			data = load_export(data_path)
		except:
			print()
			raise

		print(f' done. Found {len(data)} entries.')
		if len(data) == 0:
			print('No data found.')

	out_path = config.pull('out-path', None)

	if out_path is not None:
		out_path = Path(out_path).expanduser()
		print(f'Will save summary to {out_path}.')
	else:
		print('Will not save summary (use --out-path to save).')

	nonunique = []
	unique = set()
	for row in data:
		key = f'{row["desc_id"]}{row["given_info"]}{row["question"]}{row["answer"]}'
		if key in unique:
			nonunique.append(key)
		unique.add(key)
	if len(nonunique) > 0:
		print(f'Warning: {len(nonunique)} duplicate entries found.')

	print()
	rows = extract_summary_data(data)

	checks = sorted([k[len('contains_'):] for k in rows[0].keys() if k.startswith('contains_')])
	missing_data = np.asarray([[row.get(f'contains_{k}', False) for k in checks] for row in rows]).astype(bool)

	missing = (~missing_data).sum(0)
	if missing.sum() > 0:
		print(f'Missing fields: {dict(zip(checks, missing.tolist()))}')

	print()
	# summary by story
	_summarize_key(rows, 'story')

	print()

	# summary by query type
	_summarize_key(rows, 'query_type')
	
	key = config.pull('key', None)
	if key is not None:
		print()
		_summarize_key(rows, key)

	if out_path is not None:
		print(f'Saving summary to {out_path}')
		cols = sorted(rows[0].keys())
		df = pd.DataFrame(rows, columns=cols)
		df.to_csv(out_path, index=False)
		print('Done.')
		return out_path



def extract_summary_data(data, pbar=True):
	rows = []

	itr = data
	if pbar:
		itr = tqdm(data, desc='Analyzing questions')

	for entry in itr:
		element = {
			'story_id': entry.get('meta', {}).get('story_id', '[missing]'),
			'graph': entry.get('meta', {}).get('graph_id', '[missing]'),

			'query_type': entry.get('meta', {}).get('query_type', '[missing]'),

			'simpson': entry.get('meta', {}).get('simpson', '[missing]'),
			'sensical': entry.get('sensical', '[missing]'),

			'answer': entry.get('answer', '[missing]'),

			'contains_background': 'background' in entry,
			'contains_given_info': 'given_info' in entry,
			'contains_question': 'question' in entry,
			'contains_reasoning': 'reasoning' in entry,
			'contains_meta': 'meta' in entry,
			'contains_answer': 'answer' in entry,
			'contains_variable_mapping': 'variable_mapping' in entry.get('meta', {}),
		}
		rows.append(element)
		element['story'] = f'{element["story_id"]} - {element["graph"]}'
		
	return rows



def _summarize_key(rows, key):
	stories = {}
	for row in rows:
		stories.setdefault(row[key], []).append(row['answer'])

	summary = {story: dict(Counter(answers)) for story, answers in stories.items()}
	answer_keys = {a for story, answers in stories.items() for a in answers}

	cols = [key.capitalize(), 'Number', 'Percent', *sorted(answer_keys)]

	total = sum(map(len,stories.values()))

	tbl = [[story, len(answers), f'{len(answers)/total:.1%}', *[summary[story].get(a, 0) for a in answer_keys]]
	       for story, answers in stories.items()]

	if key == 'story':
		cols.insert(0, 'Graph')
		tbl = [[*story.split(' - ')[::-1], *row] for (story, *row) in tbl]
	
	print(f'Summary for {len(tbl)} {key.capitalize()} ({total} total questions):')
	print(tabulate(sorted(tbl, key=lambda r: r[0]), headers=cols))



@fig.script('generate')
def generate_and_store(config):
	path = config.pull('path', str(Path(config.pull('root', util.data_root(), silent=True)) / 'data.json'))
	path = Path(path)


	story_ids = config.pull('stories')
	if isinstance(story_ids, str):
		story_ids = [story_ids]
	
	queries = config.pull('queries', ())
	queries = [create_query(q) if isinstance(q, str) else q for q in queries]
	
	question_limit = config.pull('question-limit', None)  # not recommended
	
	seed = config.pull('seed', None)
	model_kwargs = config.pull('model-kwargs', {})
	detail_limit = config.pull('spec-limit', None)
	skip_det = config.pull('skip-det', False)
	graph_cap = config.pull('graph-cap', None)
	include_background = config.pull('include-background', False)
	include_reasoning = config.pull('include-reasoning', True)

	if question_limit is not None:
		print(f'WARNING: using question-limit={question_limit} is not recommended.')

	builder = config.pull('builder', None)
	if builder is None:
		builder = RandomBuilder(seed=seed)

	transformation = config.pull('transformation', None)

	overwrite = config.pull('overwrite', False)

	data = []
	if path.exists() and not overwrite:
		data = load_export(path)
		print(f'Loaded {len(data)} existing questions from {path} (will append new ones).')


	ID = len(data) + 1 + config.pull('id-offset', 0)

	separate_models = config.pull('model-meta', None)
	model_meta_list = None
	if separate_models is not None:
		separate_models = Path(separate_models)

		if separate_models.exists() and config.pull('extend-models', not overwrite):
			try:
				model_meta_list = load_export(separate_models)
			except json.JSONDecodeError:
				print(f'WARNING: could not load model meta from {separate_models}')
				model_meta_list = []
		else:
			model_meta_list = []

	solutions = {}
	
	i = 0
	for i, story_id in enumerate(story_ids):
		
		print(f'[{len(data)} questions completed] Generating questions for story: {story_id} ({i+1}/{len(story_ids)})')
		
		try:
			for entry in generate_questions(story_id, builder, transformation, queries, spec_limit=detail_limit,
			                                model_meta_list=model_meta_list, graph_cap=graph_cap,
			                                include_background=include_background, include_reasoning=include_reasoning,
			                                seed=seed, model_kwargs=model_kwargs, skip_det=skip_det):
				data.append({'question_id': ID, **entry})
				if entry['answer'] not in solutions:
					solutions[entry['answer']] = 0
				solutions[entry['answer']] += 1
				
				ID += 1
				
				if question_limit is not None and len(data) >= question_limit:
					raise KeyboardInterrupt

		except KeyboardInterrupt:
			print('Keyboard interrupt. Stopping and saving data generated so far.')
			break
	
	print('---------------------------------')
	print(f'Generated {len(data)} questions from {i+1} stories.')
	print(f'Solution profile: {solutions}')

	outpath = save_json(data, path)
	print(f'Saved to {outpath}')

	if separate_models is not None:
		save_json(model_meta_list, separate_models)
		print(f'Saved model meta to {separate_models}')

	if config.pull('summarize', True):
		return dataset_summary(config, outpath)

	return outpath



def generate_questions(story_id, builder, transformation, queries, spec_limit=None, model_meta_list=None,
                       graph_cap=None, include_background=False, include_reasoning=True,
                       seed=None, model_kwargs=None, pbar=True, skip_det=False):
	'''
	
	:param story_id: expected to be the name of a config in `config/stories/`
	:param queries: list of names of query types that should be generated (defaults to all queries in the story config)
	:param spec_limit: limit the number of distinct constraint sets that are generated (defaults to all possible)
	:param seed: for generating scm params
	:param num_samples: for MC integration to decide if queries are identifiable from the given constraints (recommended >= 10)
	:return: generator of questions (including meta-data)
	'''
	story = load_story(story_id)

	if model_kwargs is None:
		model_kwargs = {}
	
	if 'scm' not in story:
		raise ValueError(f'"scm" missing in {story}')

	if not isinstance(queries, (list, tuple)):
		queries = [queries]
	if len(queries) == 0:
		if 'queries' not in story:
			raise ValueError(f'"queries" missing in {story}')
		queries = [create_query(q) for q in story['queries']]
	# print(queries)
	# print(f'Generating questions for story: {story_id} (queries: {", ".join(q.query_name for q in queries)})')

	graphs = story.get('phenomenon', [])
	if isinstance(graphs, str):
		graphs = [graphs]
	if graph_cap is not None:
		graphs = graphs[:graph_cap]

	for graph_id in graphs:
		if skip_det and graph_id.startswith('det'):
			print(f'Skipped deterministic story: {story_id}')
			return

		# rng = np.random.default_rng(seed)

		story['phenomenon'] = graph_id

		if spec_limit is None:
			print(f'WARNING: no spec limit specified, defaulting to 1.')
			spec_limit = 1
		count = spec_limit
		if builder.is_deterministic:
			count = min(count, builder.spec_count(story))
		if count is None:
			raise NotImplementedError(f'Cannot determine number of specs for {graph_id}')

		labels = validate_story(story.copy())
		if transformation is not None:
			labels = transformation.transform(labels)

		failures = {}
		itr = enumerate(builder.sample_specs(story, count))
		if pbar:
			itr = tqdm(itr, total=count, desc=f'iterating specs for {story_id} ({graph_id})') # Iterating through distinct graph specifications
		for spec_id, spec in itr:

			model = builder.generate_scm(story, spec, seed=seed, **model_kwargs)

			model_meta = {
				'story_id': story_id,
				'graph_id': graph_id,
			              'spec_id': spec_id,
			              'spec': spec,
			              'seed': seed,
			              'builder': getattr(builder, 'name', None),
			              **builder.meta_data(model, labels, spec),
			              **labels.get('meta', {}),
			              'equation_type': getattr(model, 'equation_type', None),
			              'background': model.verbalize_background(labels),
			              'variable_mapping': model.variable_mapping(labels),
						  'structure': model.symbolic_graph_structure(),
			              'params': {str(v): v.param.tolist() for v in model.variables()},
			              **model_kwargs}
			model_id = len(model_meta_list) if model_meta_list is not None else None
			if model_meta_list is not None:
				model_meta['groundtruth'] = {
					'ATE(Y | X)': model.ate('X')['Y'],
					'ETT(Y | X)': model.ett('X')['Y'],
					'NDE(Y | X)': model.nde('X', 'Y')['Y'],
					'NIE(Y | X)': model.nie('X', 'Y')['Y'],
					'P(Y=1 | X=1)': model.marginals(X=1)['Y'],
					'P(Y=1 | X=0)': model.marginals(X=0)['Y'],
					**{f'P({k}=1)': v for k,v in model.marginals().items()}
				}
				model_meta = {'model_id': model_id, **model_meta}
				model_meta_list.append(model_meta)

			for query in queries:
				try:
					for question_id, entry in enumerate(query.generate_questions(model, labels)):
						if 'meta' not in entry:
							entry['meta'] = {}
						if model_meta_list is None:
							entry['meta']['model'] = model_meta
						else:
							entry['meta']['model_id'] = model_id
							if include_background:
								entry = {'background': model_meta['background'], **entry}

						if include_reasoning:
							entry['reasoning'] = query.reasoning(model, labels, entry)

						entry['meta'] = {
							'story_id': story_id,
							'graph_id': graph_id,
							**entry.get('meta', {})
						}

						yield {'desc_id': f'{story_id}-{graph_id}-{query.name}-'
						                    f'model{model_id}-spec{spec_id}-q{question_id}',
						         **entry}

				except QueryFailedError as e:
					if query not in failures:
						failures[query] = (query, e)

		if pbar:
			itr.close()
		if len(failures) > 0:
			print('\n'.join(f'Query {query.name!r} failed for {graph_id!r} (story {story_id!r}): {e}'
			                for query, e in failures.values()))




###################################################################################################

























