from .imports import *

from .stories import get_available_stories, iterate_scenarios, get_story_system
from .queries import create_query
from .verbalization import generate_ambiguous_evidence, \
	verbalize_ambiguous_evidence_from_value, verbalize_precise_evidence


def balanced_choose(gen, options: Sequence[str], counts: Dict[str, int] = None):
	if counts is None:
		counts = {}
	
	if isinstance(options, dict):
		keys = list(options.keys())
	else:
		keys = options
	
	wts = {x: max(1, counts.get(x, 1)) for x in options}
	total = sum(wts.values())
	wts = {x: total / w for x, w in wts.items()}
	
	pick = gen.choice(keys, p=[wts[x] for x in options])
	if isinstance(options, dict):
		return options[pick]
	else:
		return pick



def update_history(q, history):
	story_name = q['story_id']
	
	if story_name not in history['stories']:
		history['stories'][story_name] = 0
	history['stories'][story_name] += 1
	


def generate_ate_question(N=100, stories=None, *, history=None, seed=None, commonsense=None,
                          implicit_edges=None, relative_probs=None, ambiguous_evidence=None, dropped_evidence=None):
	if history is None:
		history = {}
	
	gen = np.random.RandomState(seed)
	
	stories = stories or list(get_available_stories())
	assert len(stories) > 0, "No stories available"
	
	full = [list(iterate_scenarios(story)) for story in stories]
	story_options = {}
	for scens in full:
		if scens[0]['query'] == 'ate':
			assert len(scens) == 2
			story_options[scens[0]['story_id']] = scens[0], scens[1]
			story_options[scens[1]['story_id']] = scens[1], scens[0]
	
	operator = create_query('ate')
	
	completed = 0
	while completed < N:
		winner, loser = balanced_choose(gen, story_options, history.setdefault('stories', {}))
		
		q = {
			# 'story': story_name,
			# 'dof': ,
		}
		
		# pick fixed
		
		

		param1, param2 = winner['commonsense'], loser['commonsense']
		
		ID = winner['story_id']
		name = winner['name']
		key = winner['scenario']
		# query = winner['query']
		
		# if skip_existing and 'commonsense' in story:
		# 	print(f'({i + 1}/{len(stories)}) Skipping {ID}')
		# 	print('-' * 60)
		# 	continue
		
		# system = get_story_system(winner)
		operator = create_query('ate', system=system)
		
		# prompts.append({
		# 	'ID': ID,
		# 	'story': name,
		# 	'scenario': key,
		# 	'query': query,
		# 	'graph': story['graph'],
		# 	'variables': story['variables'],
		# 	'keys': operator.prompt_keys(system, story),
		# 	'prompt': prompt_builder(operator, system, story),
		# })
		
		update_history(q, history)
		yield q








