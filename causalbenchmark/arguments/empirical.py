from .imports import *
from .queries import create_query
from .stories import iterate_scenarios, get_story_system, get_available_stories


def prompt_variables(story):
	vars = story.get('variables')
	assert vars is not None
	
	lines = []
	for v, r in vars.items():
		lines.append(f'{r}=1: {story.get(v + "=1")}')
		lines.append(f'{r}=0: {story.get(v + "=0")}')
	
	return '\n'.join(lines)


prompt_template = '''We are conducting an investigation into how well Large Language Models do commonsense reasoning. Imagine a simple causal graph of three binary variables:

{variables}

Propose some reasonable ranges (including an upper and lower bound for the 90% confidence interval) of marginal and conditional probabilities for the following quantities:

{params}

Please respond only by replacing the "?" with the range.'''


def prompt_builder(query, system, story):
	vars = prompt_variables(story)
	
	params = query.prompt_params(system, story)
	
	prompt = prompt_template.format(variables=vars, params=params)
	
	return prompt


@fig.script('param-prompts')
def param_prompts(config):
	outpath = config.pull('out', str(util.data_root() / 'prompts.json'))
	if outpath is not None:
		outpath = Path(outpath)
		print(f'Writing prompts to {outpath}')
	
	skip_existing = config.pull('skip-existing', False)
	
	story_names = config.pull('stories')
	
	dirroot = config.pull('stories-root', None)
	stories = [story for name in story_names for story in iterate_scenarios(name, root=dirroot)]
	
	prompts = []
	
	print('-' * 60)
	
	for i, story in enumerate(stories):
		name = story['name']
		key = story['scenario']
		query = story['query']
		ID = f'{name}-{key}-{query}'
		
		if skip_existing and 'commonsense' in story:
			print(f'({i + 1}/{len(stories)}) Skipping {ID}')
			print('-' * 60)
			continue
		
		system = get_story_system(story)
		operator = create_query(query, system=system)
		
		prompts.append({
			'ID': ID,
			'story': name,
			'scenario': key,
			'query': query,
			'graph': story['graph'],
			'variables': story['variables'],
			'keys': operator.prompt_keys(system, story),
			'prompt': prompt_builder(operator, system, story),
		})
		
		print(f'({i + 1}/{len(stories)}) Created prompt {ID}')
		print('-' * 60)
		print(prompts[-1]['prompt'])
		print('-' * 60)
	
	# print('-' * 60)
	if outpath is None:
		print(f'Created {len(prompts)} prompts')
	else:
		print(f'Saving {len(prompts)} prompts')
		
		save_json(prompts, outpath)
		
		print(f'Saved {len(prompts)} prompts to {outpath}')
	
	return prompts
