from pathlib import Path
from omnibelt import save_json, load_json, load_yaml
import omnifig as fig

from .. import util
from .systems import create_system


role_keys = {
	'treatment': 'X',
	'outcome': 'Y',
	'confounder': 'U',
	'instrument': 'Z',
	'mediator': 'M',
}



def iterate_scenarios(base):
	if isinstance(base, (Path, str)):
		base = load_yaml(base)
	for key, scenario in base.get('scenarios', {'base': base}).items():
		story = base.copy()
		story.update(scenario)
		story['scenario'] = key
		story['variables'] = {variable: role_keys[role] for role, variable in story.get('roles', {}).items()}
		for query in story.get('queries', []):
			yield {'query': query, **story}



def prompt_variables(story):
	vars = story.get('variables')
	assert vars is not None
	
	lines = []
	for v, r in vars.items():
		lines.append(f'{r}=1: {story.get(v + "=1")}')
		lines.append(f'{r}=0: {story.get(v + "=0")}')
	
	return '\n'.join(lines)



class Prompter:
	@staticmethod
	def prompt_keys(system, story):
		raise NotImplementedError
	
	
	@classmethod
	def prompt_params(cls, system, story):
		keys = cls.prompt_keys(system, story)
		
		lines = []
		for key in keys:
			lines.append(f'p({key}) = ?')
		
		return '\n'.join(lines)



class ATE_Prompter(Prompter):
	@staticmethod
	def prompt_keys(system, story):
		dofs = system.ate_dofs
		
		keys = []
		for dof in dofs:
			v, *other = dof.split('|')
			v = v + '=1'
			if len(other):
				v = v + '|' + '|'.join(other)
			keys.append(v)
			
		return keys
	


class Mediation_Prompter(Prompter):
	@staticmethod
	def prompt_keys(system, story):
		return list(set(system.nde_dofs) | set(system.nie_dofs))


	
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
	outpath = Path(outpath)
	print(f'Writing prompts to {outpath}')
	
	skip_existing = config.pull('skip-existing', True)
	
	stories = config.pull('stories')
	
	queries = {'ate': ATE_Prompter(), 'med': Mediation_Prompter()}
	
	dirroot = config.pull('stories-root', str(util.assets_root()))
	dirroot = Path(dirroot)
	
	prompts = []
	
	for name in stories:
		story_path = dirroot / f'{name}.yml'
		for story in iterate_scenarios(story_path):
			if skip_existing and 'commonsense' in story:
				print(f'Skipping {name}-{story["scenario"]}-{story["query"]}')
				print('-'*60)
				continue
			
			key = story['scenario']
			query = story['query']
			ID = f'{name}-{key}-{query}'
			
			system = create_system(story['graph'])
			
			prompts.append({
				'ID': ID,
				'story': name,
				'scenario': key,
				'query': query,
				'graph': story['graph'],
				'variables': story['variables'],
				'keys': queries[query].prompt_keys(system, story),
				'prompt': prompt_builder(queries[query], system, story),
			})
			
			print(f'Created prompt {ID}')
			print('-'*60)
			print(prompts[-1]['prompt'])
			print('-'*60)
		
	# print('-' * 60)
	print(f'Saving {len(prompts)} prompts')
	
	save_json(prompts, outpath)
	
	print(f'Saved {len(prompts)} prompts to {outpath}')



