from .imports import *



class Verbalizer:
	@staticmethod
	def load_templates(path=None):
		if path is None:
			path = util.assets_root() / 'templates.yml'
		assert path.exists(), f'path {path!r} does not exist'
		return load_yaml(path)


	@classmethod
	def load_from_path(cls, path):
		assert path.exists(), f'path {path!r} does not exist'
		return cls(load_yaml(path))

	_Selector = None


	def __init__(self, data=None):
		self.data = data



class _Selector(Verbalizer):
	def __init__(self, verbalizer, value):
		self.verbalizer = verbalizer
		self.value = value
Verbalizer._Selector = _Selector



class StoryVerbalizer(Verbalizer):
	def __init__(self, data):
		super().__init__(data)
		self.vars = {key: VariableVerbalizer(value) for key, value in data.get('variables', {}).items()}


	def find(self, vname, key, value=None):
		return self.vars[vname].find(key, value=value)


	def get_variable(self, var_name, value=None):
		if value is None:
			return self.vars[var_name]
		return self.vars[var_name].with_value(value)


	def get_term(self, term):
		vname, value, parents = TermVerbalizer.parse_term(term)
		var = self.get_variable(vname)
		if value is None:
			assert var.N == 2, f'variable {vname!r} has {var.N} values, but no value was specified'
			value = '1'
		var = var.with_value(value)
		conds = [self.get_variable(k, value=v) for k, v in parents.items()]
		return TermVerbalizer(var, conditions=conds)



class VariableVerbalizer(Verbalizer):
	def __init__(self, data, value=None):
		self.data = data
		self.value = value
		self.categories = self.data.get('categories', [str(v) for v in self.data.get('values', {}).keys()
		                                               if not str(v).startswith('~')])
		# self.values = {self.parse_variable_value(key): value for key, value in self.data.get('values', {}).items()}

		templates = self.load_templates()
		self.default_structures = templates.get('default-structures', {})


	def parse_variable_value(self, value):
		if isinstance(value, int):
			value = str(value)
		if value.startswith('~'):
			value = tuple(sorted([c for c in self.categories if c != value[1:]]))
		return value


	@property
	def N(self):
		return len(self.categories)


	def find(self, key):
		'''
		:param key: subject, verb, do, name, unit, pronoun, event, head, status, ...
		:param value:
		:return:
		'''
		if value is not None:
			base = self.values.get(self._parse_variable_value(value), {})
			if key in base:
				return base[key]
		if key in self.data:
			return self.data[key]
		if key in self.default_structures:
			return self.fill_in(self.default_structures[key])
		raise KeyError(key)



	@staticmethod
	def extract_sentence_keys(template):
		for match in re.finditer(r'\{([^\}]+)\}', template):
			yield match.group(1)


	def fill_in(self, template):
		reqs = list(self.extract_sentence_keys(template))



		pass



class VariableValueVerbalizer(Verbalizer):



	pass



class TermVerbalizer(Verbalizer):
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


	def __init__(self, var, value, conditions=None):
		self.var = var
		self.value = value
		self.given = conditions or {}


	def find(self, key):
		return self.var.find(key)



class Template:
	_template_key = None
	@classmethod
	def load_root(cls, path=None):
		if path is None:
			path = util.assets_root() / 'templates.yml'
		assert path.exists(), f'path {path!r} does not exist'
		return cls(load_yaml(path))


	def __init__(self, data, *, seed=None):
		if self._template_key is not None and self._template_key in data:
			info = data[self._template_key]
		else:
			info = data
		self.info = info
		self.gen = np.random.RandomState(seed)
		self.full_data = data



class PremiseTemplate(Template):
	pass



class FrequencyPremise(PremiseTemplate):
	_template_key = 'frequency'
	def __init__(self, data, **kwargs):
		super().__init__(data, **kwargs)
		self.sentences = self.info['structure']
		self.options = self.info['options']
		self.option_keys = list(self.options.keys())
		self.unique = [(index, key) for index, key in product(range(len(self.sentences)), self.options.keys())]


	@staticmethod
	def random_order(gen: np.random.RandomState, options):
		yield from gen.permutation(options)


	def generate(self, verbalizer):
		# var, value, parents = self.parse_term(term)

		for index, key in self.random_order(self.gen, self.unique):
			index = int(index)
			template = self.sentences[index]
			reqs = list(self.extract_sentence_keys(template))

			details = self.options[key]

			entries = {}
			for req in reqs:
				if req in details:
					entries[req] = details[req]
				else:
					entries[req] = verbalizer.find(req)

			if len(entries) != len(reqs):
				continue
			line = template.format(**entries)

			yield {'verb': line, 'implication': details['implication'], 'ID': f'{key}-{index}'}



def test_freq():
	story_path = util.assets_root() / 'arguments' / 'contagion.yml'
	story = load_yaml(story_path)

	base = StoryVerbalizer(story)

	term = 'Y|U=0,X=1'
	verbalizer = base.get_term(term)

	style = FrequencyPremise.load_root()

	premises = []
	for premise in style.generate(verbalizer):
		premises.append(premise)

	print(premises)

	pass











