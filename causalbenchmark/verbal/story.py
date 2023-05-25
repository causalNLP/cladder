

from ..graphs import get_graph_class
from .. import util



def _replace_keys(data, fixes):
	if isinstance(data, list):
		return [_replace_keys(v, fixes) for v in data]
	elif isinstance(data, dict):
		return {fixes.get(k, k): _replace_keys(v, fixes) for k, v in data.items()}
	return data


_node_corrections = {
		# X->V2,V2->Y
		"chain": {'X': 'X', 'V2': 'Z', 'Y': 'Y'},
		# X->V3,Y->V3
		"collision": {'X': 'X', 'Y': 'Y', 'V3': 'Z'},
		# V1->X,V1->Y,X->Y
		"confounding": {'V1': 'Z', 'X': 'X', 'Y': 'Y'},
		# X->V2,X->Y,V2->Y
		"mediation": {'X': 'X', 'V2': 'Z', 'Y': 'Y'},
		# V1->X,X->V3,V1->Y,V3->Y
		"frontdoor": {'V3': 'Z', 'V1': 'W', 'X': 'X', 'Y': 'Y'},
		# V1->X,V2->X,V1->Y,X->Y
		"IV": {'V2': 'Z', 'V1': 'W', 'X': 'X', 'Y': 'Y'},
		# X->V3,V2->V3,X->Y,V2->Y,V3->Y
		"arrowhead": {'X': 'X', 'V3': 'Z', 'V2': 'W', 'Y': 'Y'},
		# X->V3,X->V2,V2->Y,V3->Y
		"diamond": {'X': 'X', 'V2': 'Z', 'V3': 'W', 'Y': 'Y'},
		# V1->V3,V1->X,X->Y,V3->Y
		"diamondcut": {'V1': 'Z', 'V3': 'W', 'X': 'X', 'Y': 'Y'},

		'fork': {'X': 'X', 'V2': 'Z', 'Y': 'Y'},
	}


def validate_story(raw):
	if 'phenomenon' in raw: # fix labels based on node corrections

		phen = raw['phenomenon']

		if isinstance(phen, str) and phen in _node_corrections:
			corrections = {v: k for k, v in _node_corrections[phen].items()}
		# else:
		#
		# base = get_graph_class(phen, None)
		#
		# corrections = {v.name: f'V{i+1}' for i, v in enumerate(base.static_variables())
		#                if v.name not in {'X', 'Y'}}

			fixes = {}
			for old, new in corrections.items():
				for k, v in raw.items():
					if k.startswith(old):
						if isinstance(v, float):
							v = 'confounder active' if '1' in k else 'confounder inactive'
						fixes[new + k[len(old):]] = v
			raw.update(fixes)


			if 'easy' in raw:
				raw['easy'] = _replace_keys(raw['easy'], corrections)
			if 'hard' in raw:
				raw['hard'] = _replace_keys(raw['hard'], corrections)

		# graph = util.load_graph_config(phen)
		# corrections = graph.get('node_corr', {})
		# fixes = {}
		# for new, old in corrections.items():
		# 	for k, v in raw.items():
		# 		if k.startswith(old):
		# 			fixes[new + k[len(old):]] = v
		# raw.update(fixes)

		# old2new = {v: k for k, v in corrections.items()}
	return raw



def load_story(story_id):
	path = util.story_root() / f'{story_id}.yml'
	if not path.exists():
		return {}
	story = util.load_yaml(path)
	return validate_story(story)








