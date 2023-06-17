from pathlib import Path
from omnibelt import load_yaml, unspecified_argument
import omnifig as fig

from .. import util
from ..util import Seeded, Parameterized, hparam



class StoryTransformation:
	def transform(self, story):
		raise NotImplementedError


@fig.component('anticommonsense')
class AnticommonsenseTransformation(StoryTransformation, Seeded):
	@hparam
	def path(self):
		return util.assets_root() / 'anticommonsense.yml'
	@path.formatter
	def path(self, value):
		self._options = None
		return Path(value)


	@hparam
	def options(self):
		return load_yaml(self.path)


	def transform(self, story):
		sub_options = self.options.get('subjects', {})
		var_options = self.options.get('variables', {})

		vars = [k[:-len('subject')] for k,v in story.items() if k.endswith('subject')]
		vars = [v for v in vars if v in var_options]

		picks = [(v, option) for v in vars for option in sub_options[v].get(story.get(f'{v}subject'), [])]

		if len(picks) == 0:
			raise NotImplementedError(story['story_id'])

		var, choice = self._rng.choice(picks)

		subject = story.get(f'{var}subject')
		plural = self.options.get('plurals', {}).get(subject, f'{subject}s')

		fixes = var_options[var][choice]
		fixes = {k: util.pformat(v, subject=subject, plural=plural, **story)
		          for k, v in fixes.items() if k in story}

		change = story.copy()
		change.update(fixes)
		change['meta'] = {'anticommonsense': {'variable': var, 'change': choice}, **change.get('meta', {})}

		return change


























