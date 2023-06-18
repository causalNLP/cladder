from pathlib import Path
from omnibelt import save_json, load_json, load_yaml

from .. import util
from .systems import create_system


role_keys = {
	'treatment': 'X',
	'outcome': 'Y',
	'confounder': 'U',
	'instrument': 'Z',
	'mediator': 'M',
}



def story_root():
	return util.assets_root() / 'new'



def get_available_stories(root=None):
	if root is None:
		root = story_root()
	
	for path in root.glob('**/*.yml'):
		# yield path.relative_to(root)
		yield path.stem



def get_story_system(story):
	return create_system(story['graph'])



def iterate_scenarios(path, root=None):
	if root is None:
		root = story_root()
	
	path = Path(path)
	if not path.exists():
		if path.suffix == '':
			path = path.with_suffix('.yml')
		path = root / path
	assert path.exists()
	base = load_yaml(path)
	for key, scenario in base.get('scenarios', {'base': base}).items():
		for query in base.get('queries', []):
			story = {**base, **scenario}
			yield {'query': query,
			       'path': str(path),
			       'name': path.stem,
			       'scenario': key,
			       'variables': {variable: role_keys[role] for role, variable in story.get('roles', {}).items()},
			       **story}



def get_all_stories():
	stories = sorted(get_available_stories())
	full = [list(iterate_scenarios(story)) for story in stories]
	return full