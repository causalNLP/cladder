
from .imports import *
from omniply.novo.test_novo import *



class Template:
	pass


class Atom(AbstractTool):

	'''selects between static options'''
	def possibilities(self):
		raise NotImplementedError

	pass


class TemplateSelector(Template):
	'''selects between (tools)'''
	pass


class StaticTemplater(Atom):
	def __init__(self, key: str, template: str):
		self.key = key
		self.template = template


	def gizmos(self) -> Iterator[str]:
		yield self.key


	def _extract_keys(self, template):
		for match in re.finditer(r'\{([^\}]+)\}', template):
			yield match.group(1)


	def grab_from(self, ctx: Optional['AbstractContext'], gizmo: str) -> Any:
		reqs = {key: ctx.grab_from(ctx, key) for key in self._extract_keys(self.template)}
		return self.template.format(**reqs)



class Decision(Atom):
	@property
	def name(self):
		raise NotImplementedError
	@property
	def num_options(self):
		raise NotImplementedError
	@property
	def options(self):
		raise NotImplementedError
	def random_choice(self, gen: np.random.RandomState):
		return gen.choice(list(self.options))


class Verbalizer(Context, LoopyKit, MutableKit):
	def __init__(self, *args, identity=None, seed=None, **kwargs):
		super().__init__(*args, **kwargs)
		if identity is None:
			identity = {}
		self.identity = identity
		self.gen = np.random.RandomState(seed)


	def identify(self, decision: Decision):
		name = decision.name
		if name not in self.identity:
			self.identity[name] = decision.random_choice(self.gen)
		return self.identity[name]


	# _default_templater = StaticTemplater
	# def package(self, val: Any, gizmo: str = None):
	# 	if isinstance(val, str): # recursively fill in templates
	# 		return self._default_templater(gizmo, val).grab_from(self, gizmo)
	# 	# raise NotImplementedError
	# 	return val



class StaticChoice(Decision):
	def __init__(self, name: str, data: Dict[str, Union[str, Dict[str, Any]]]):
		self._name = name
		self.data = data


	@property
	def name(self):
		return self._name


	def _parse_data(self, data):
		raise NotImplementedError


	@property
	def num_options(self):
		return len(self.data.keys())
	@property
	def options(self):
		yield from self.data.keys()


	def gizmos(self) -> Iterator[str]:
		yield from filter_duplicates([self.name], *[val for val in self.data.values() if isinstance(val, dict)])


	def _default_value(self, ctx, ID, gizmo):
		if gizmo in self.data:
			return self.data[gizmo]
		raise NotImplementedError


	def grab_from(self, ctx: Verbalizer, gizmo: str) -> Any:
		ID = ctx.identify(self)
		if gizmo not in self.data[ID]:
			return self._default_value(ctx, ID, gizmo)
		return self.data[ID][gizmo] # TODO: should probably be done by the verbalizer automatically



class TemplateChoice(StaticChoice):
	def __init__(self, name: str, data: Dict[str, Union[str, Dict[str, Any]]]):
		super().__init__(name, data)
		self.templates = {key: StaticTemplater(self.name, val if isinstance(val, str) else val[self.name])
		                  for key, val in self.data.items()}


	def grab_from(self, ctx: Verbalizer, gizmo: str) -> Any:
		if gizmo == self.name:
			return self.templates[ctx.identify(self)].grab_from(ctx, gizmo)
		return super().grab_from(ctx, gizmo)



class Sourced:
	def __init__(self, data=None):
		if data is None:
			data = {}
		self.data = data



class Story(Sourced):
	@staticmethod
	def parse_term(term):
		var, *given = term.split('|')
		parents = None
		if len(given):
			parents = {k: str(v) for k, v in [x.split('=') for x in given[0].split(',')]}

		value = None
		if '=' in var:
			var, value = var.split('=')
			value = int(value)

		return var, value, parents


	@classmethod
	def set_term_value(cls, term, value=None, parents=None):
		var, old_value, old_parents = cls.parse_term(term)

		if parents is None:
			parents = old_parents
		parents = ','.join(f'{k}={v}' for k, v in parents.items()) if parents is not None else ''

		head = f'{var}={value}' if value is not None else var
		tail = f'|{parents}' if len(parents) else ''

		return head + tail


	def variable(self, name: str, value=None):
		return Variable(self.data['variables'][name], value=value)


	def term(self, term: str):
		var, value, parents = self.parse_term(term)
		conditions = {k: self.variable(k, value=v) for k, v in parents.items()}
		return StatisticalTerm(self.data['variables'][var], conditions=conditions, value=value)



class Variable(Sourced, AbstractTool):
	def __init__(self, data, value=None):
		super().__init__(data)
		if value is None:
			value = self.data.get('default', '1')
		self.categories = self.data.get('categories', [str(v) for v in self.data.get('values', {}).keys()
		                                               if not str(v).startswith('~')])
		self.value = self.parse_variable_value(value)
		values = {str(k): v for k, v in self.data.get('values', {}).items()}
		self.content = values.get(self.value, {})


	def gizmos(self) -> Iterator[str]:
		yield from filter_duplicates(self.content.keys(), (key for key in self.data.keys() if key != 'values'))


	def parse_variable_value(self, value):
		if isinstance(value, int):
			value = str(value)
		if value.startswith('~'):
			value = tuple(sorted([c for c in self.categories if c != value[1:]]))
		return value


	@property
	def N(self):
		return len(self.categories)


	def grab_from(self, ctx: Verbalizer, gizmo: str) -> Any:
		if gizmo in self.content:
			return self.content[gizmo]
		if gizmo in self.data:
			return self.data[gizmo]
		raise NotImplementedError



class StatisticalTerm(Variable):
	def __init__(self, data, *, conditions=None, **kwargs):
		super().__init__(data, **kwargs)
		self.given = conditions or {}



def default_vocabulary(seed=None):
	verb = Verbalizer(seed=seed)

	full = _get_template_data()

	term_defaults = full['default-structure']
	terms = [StaticTemplater(key, val) for key, val in term_defaults.items()]
	verb.include(*terms)

	freq = TemplateChoice('freq_text', full['frequency']['options'])
	verb.include(freq)

	event = TemplateChoice('event_text', full['event-structure'])
	verb.include(event)

	return verb



def _get_template_data():
	path = util.assets_root() / 'templates.yml'
	assert path.exists(), f'path {path!r} does not exist'
	return load_yaml(path)


def test_templater():
	seed = 0
	ctx = default_vocabulary(seed=seed)

	known = list(ctx.gizmos())

	out = ctx['freq_text']
	impl = ctx['implication']
	id_info = ctx.identity

	print(out)


def test_story():
	seed = 0
	ctx = default_vocabulary(seed=seed)

	story_data = load_yaml(util.assets_root() / 'arguments' / 'contagion.yml')
	story = Story(story_data)

	term = 'Y|U=0,X=0'
	verb = story.term(term)
	ctx.include(verb)

	out = ctx['freq_text']

	phase = ctx['phase']


	print(out)

	pass

