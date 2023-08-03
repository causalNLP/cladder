import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from tabulate import tabulate
from omnibelt import save_yaml, load_yaml, load_export, export, save_json, load_json, unspecified_argument
import yaml, io
from yaml.parser import ParserError
import omnifig as fig
import pandas as pd

from .generator import dataset_summary
from . import util


class SkipItem(Exception):
	pass

def _extract_key(entry, models):

	meta = entry.get('meta', {})

	model = models.get(entry.get('model_id', None), {})

	story_id = meta.get('story_id', None)
	graph_id = meta.get('graph_id', None)
	query_type = meta.get('query_type', None)
	answer = entry.get('answer', None)

	if query_type == 'correlation' and graph_id == 'collision':
		raise SkipItem


	return (
		meta.get('story_id', None),
        meta.get('graph_id', None),
        meta.get('query_type', None),
        entry.get('answer', None),
        model.get('difficulty', None),
        # 'not-anti' if model.get('anticommonsense', None) is None else 'anticommonsense',
        # 'nonsense' if meta.get('story_id', '').startswith('nonsense') else 'not-nonsense',
	        )


def _extract_sensicalness_key(entry, models):
	meta = entry.get('meta', {})

	model = models.get(entry.get('model_id', None), {})

	story_id = meta.get('story_id', None)
	graph_id = meta.get('graph_id', None)
	query_type = meta.get('query_type', None)
	answer = entry.get('answer', None)

	if query_type == 'correlation' and graph_id == 'collision':
		raise SkipItem

	return (
		'anti' if 'anticommonsense' in model else ('non' if model.get('nonsense', False) else 'common'),
		# meta.get('story_id', None),
		meta.get('graph_id', None),
		meta.get('query_type', None),
		entry.get('answer', None),
		model.get('difficulty', None),
		# 'not-anti' if model.get('anticommonsense', None) is None else 'anticommonsense',
		# 'nonsense' if meta.get('story_id', '').startswith('nonsense') else 'not-nonsense',
	)


def find_gold(options, num, rng):

	selected = []

	remaining = {}

	num = min(num, min(len(uids) for uids in options.values()))
	assert num > 0

	for combo, uids in options.items():# tqdm(options.items(), total=len(options), desc='balancing'):
		sel = rng.choice(uids, num, replace=False)
		unsel = [uid for uid in uids if uid not in sel]
		if len(unsel):
			remaining[combo] = unsel
		selected.extend(sel)

	return selected, remaining


@fig.script('merge')
def merge_and_balance(config):

	rng = np.random.RandomState(config.pull('seed', 0))

	out_path = Path(config.pull('path', 'data/merged.json'))

	# out_model_path = Path(config.pull('out-model-path', 'data/merged-models.json'))
	# print(f'Will save merged data to {out_path} (and models to {out_model_path})')

	data_files = config.pull('paths', None)
	if data_files is None:
		raise ValueError('no data files specified')

	data_files = [Path(p) for p in data_files]

	assert all(p.exists() for p in data_files), 'not all files exist'

	print(f'Found {len(data_files)} data files: {[p.stem for p in data_files]}')

	model_path = config.pull('model-path', None)
	if model_path is None:
		raise NotImplementedError
	model_path = Path(model_path)

	assert model_path.exists(), 'model file does not exist'

	datasets = [load_export(p) for p in data_files]

	models = load_export(model_path)

	print(f'Loaded {len(models)} models from {model_path}')
	print(f'Loaded {len(datasets)} data files: {[len(d) for d in datasets]}')

	full = [entry for data in datasets for entry in data]
	for i, entry in enumerate(full):
		entry['_uid'] = i

	model_table = {info['model_id']: info for info in models}
	assert len(model_table) == len(models), 'duplicate model ids'

	extract_fn = _extract_sensicalness_key if config.pull('sensicalness', False) else _extract_key

	budget = config.pull('budget', 10000)

	stats = {}
	for entry in tqdm(full):
		try:
			stats.setdefault(extract_fn(entry, model_table), []).append(entry['_uid'])
		except SkipItem:
			pass

	least = min(len(v) for v in stats.values())
	rare = [k for k, v in stats.items() if len(v) == least]
	print(f'{len(rare)} Least common combos have {least} entries')

	easy = len(stats)*least

	if budget is None:
		print(f'No cap specified, using {easy} entries')
	elif budget > len(full):
		raise ValueError(f'Cap ({budget}) is larger than the number of entries ({len(full)})')

	print(f'Balancing trivially with {easy} entries available.')

	unique = least if budget is None else int(np.ceil(budget / len(stats)).item())

	total = budget if budget is not None else easy

	selected = []
	remaining = stats

	rounds = 0

	while len(selected) < total:
		least = min(len(v) for v in remaining.values())
		num = least if budget is None else min(int((np.ceil(budget-len(selected)) / len(remaining)).item()), least)
		sel, remaining = find_gold(remaining, max(1,num), rng)
		selected.extend(sel)
		rounds += 1
		if budget is None:
			break

	print(f'Balancing took {rounds} to get {len(selected)} entries')

	selected = sorted(selected)

	gold = [full[i] for i in selected]
	for i, g in enumerate(gold):
		g['question_id'] = g.pop('_uid')

	if config.pull('summarize', True):
		summary = dataset_summary(config, data=gold)

	save_json(gold, out_path)

	print(f'Saved {len(gold)} entries to {out_path}')

	return gold



def _yamlify_csv_rows(full_data):
	for i, row in full_data.iterrows():

		data = {}
		for k, v in row.to_dict().items():
			if not k.startswith('Unnamed:'):
				try:
					data[k] = yaml.load(io.StringIO(v), Loader=yaml.SafeLoader) if isinstance(v, str) else v
				except ParserError:
					data[k] = v

		yield data



@fig.script('sync-csv')
def csv2config(config):
	raise NotImplementedError('not supported anymore')

	csvdir = config.pull('csvdir', str(util.repo_root() / 'causalbenchmark' / 'verbal'))
	csvdir = Path(csvdir)

	csvname = config.pull('csvname', '*')
	todo = list(csvdir.glob(f'{csvname}.csv'))

	if len(todo) == 0:
		raise ValueError(f'No CSV files found: {csvdir}/{csvname}.csv')

	print(f'Found {len(todo)} CSV files in {csvdir}: {[csv.stem for csv in todo]}')

	todo = {csv.stem: csv for csv in todo}

	configroot = util.config_root()

	outdir = config.pull('outdir', str(configroot / 'new'))
	outdir = Path(outdir)

	print(f'Writing configs to {outdir}')
	outdir.mkdir(parents=True, exist_ok=True)

	if 'phenomenon2graph' in todo:
		(outdir / 'phenomena').mkdir(parents=True, exist_ok=True)

		full_data = pd.read_csv(todo['phenomenon2graph'])

		for data in tqdm(_yamlify_csv_rows(full_data), total=len(full_data), desc='phenomena2graph'):
			if 'phenomenon' in data:
				phen = data['phenomenon']
				save_yaml(data, outdir / 'phenomena' / f'{phen}.yml')

			else:
				raise ValueError(f'missing phenomenon: {data}')

		del todo['phenomenon2graph']


	if 'graph2story' in todo:
		(outdir / 'stories').mkdir(parents=True, exist_ok=True)

		full_data = pd.read_csv(todo['graph2story'])

		for data in tqdm(_yamlify_csv_rows(full_data), total=len(full_data), desc='graph2story'):
			if 'story_id' in data:
				story_id = data['story_id']
				save_yaml(data, outdir / 'stories' / f'{story_id}.yml')

			else:
				raise ValueError(f'missing graph: {data}')

		del todo['graph2story']


	# TODO: Add the rest of the CSV files


	if len(todo):
		print(f'Skipping {len(todo)} CSV files: {list(todo.keys())}')


	pass






















