from typing import Tuple, List
import sys, os
from pathlib import Path
from omnibelt import load_yaml, unspecified_argument
import omnifig as fig
from string import Formatter

from argparse import Namespace

from omniply import Parameterized as _Parameterized, hparam

from itertools import product, chain, combinations

import numpy as np



vocabulary = type('Vocab', (), {
	'incdec': ['increase', 'decrease'],
	'incdecs': ['increases', 'decreases'],

	'morles': ['more', 'less'],
	'smalar': ['smaller', 'larger'],
	'smabig': ['smaller', 'bigger'],
})


def repo_root():
	return Path(__file__).parent.parent


def data_root():
	return repo_root() / 'data'


def assets_root():
	return repo_root() / 'assets'


def story_root():
	return assets_root() / 'stories'



def parse_given_info(info):

	terms = []

	for key, vals in info.items():

		var, parents = parse_mechanism_str(key)
		vals = np.asarray(vals)

		for combo, val in zip(generate_all_bit_strings(len(parents)), vals.flatten()):
			cond = dict(zip(parents, combo.tolist()))
			cond_str = ', '.join(f'{p}={v}' for p, v in cond.items())

			ant = f'P({var}=1 | {cond_str})' if len(cond) else f'P({var}=1)'

			terms.append(ant + f' = {val:.2f}')

	return terms



def parse_mechanism_str(raw: str) -> Tuple[str, List[str]]:
	term = raw.strip().replace('P(', '').replace('p(', '').split(')')[0].strip()
	var, parents = term.split('|') if '|' in term else [term, None]
	if parents is None:
		return var.strip(), []
	parents = [p.strip() for p in parents.split(',')]
	return var.strip(), parents



def generate_all_bit_strings(n, dtype=int):
	"""
	Generate all bit strings of length n as numpy bool arrays
	:param n: length of bit strings
	:return: generator of bit strings
	"""
	return np.array(list(product([0, 1], repeat=n)), dtype=dtype) # shape (2**n, n)



def powerset(iterable):
	"""
	Generate the powerset of an iterable
	:param iterable: iterable
	:return: generator of subsets
	"""
	elements = list(iterable)
	yield from chain.from_iterable(combinations(elements, r) for r in range(len(elements)+1))



def expression_format(s, **vars):
	"""
	Evaluates the keys in the given string as expressions using the given variables
	"""
	fmt = Formatter()
	vals = {key:eval(key, vars) for _, key, _, _ in fmt.parse(s)}
	return s.format(**vals)



class PowerFormatter(Formatter):
	# TODO: partial formatting - only format fields that are specified, and leave others as is
	def get_field(self, field_name, args, kwargs):
		try:
			return super().get_field(field_name, args, kwargs)
		except: # TODO: find the right exception
			return eval(self.vformat(field_name, args, kwargs), kwargs), field_name
			# return f'{{{field_name}}}', field_name


	def parse(self, s):
		start_idx = -1
		escaped = ''
		pre_idx = 0
		counter = 0
		idx = 0

		while idx < len(s):
			open_idx = s.find("{", idx)
			close_idx = s.find("}", idx)

			if open_idx == -1 and close_idx == -1:
				if counter == 0:
					# raise StopIteration
					# print(f'ending with: {escaped + s[idx:]!r}')
					yield escaped + s[idx:], None, '', None
				else:
					raise ValueError("Mismatched '{' at index {}".format(start_idx))
				break

			if open_idx != -1 and (open_idx < close_idx or close_idx == -1):
				if counter == 0:
					# yield (s[idx:open_idx], None)
					start_idx = open_idx
					pre_idx = idx
				idx = open_idx + 1
				counter += 1

			if close_idx != -1 and (close_idx < open_idx or open_idx == -1):
				if counter == 0:
					raise ValueError("Mismatched '}' at index {}".format(close_idx))
				counter -= 1
				if counter == 0:
					pre = s[pre_idx:start_idx]
					field = s[start_idx + 1:close_idx]
					if field.startswith("{") and field.endswith("}"):
						escaped = pre + '{'

						for lit, field, spec, conv in self.parse(field[1:-1]):
							if escaped is not None:
								lit = escaped + lit
								escaped = None
							yield lit, field, spec, conv

						escaped = '}'

					else:
						# spec = None
						lim = field.rfind('}')
						conv_idx = field[lim+1:].find('!')
						if conv_idx != -1:
							conv = field[lim+2+conv_idx:]
							field = field[:lim+1+conv_idx]
						else:
							conv = None

						if conv is None:
							lim = field.rfind(']')
							spec_idx = field[lim+1:].find(':')
							if spec_idx != -1:
								spec = field[lim+2+spec_idx:]
								field = field[:lim+1+spec_idx]
							else:
								spec = ''
						else:
							spec_idx = conv.find(':')
							if spec_idx != -1:
								spec = conv[spec_idx+1:]
								conv = conv[:spec_idx]
							else:
								spec = ''

						# print(f'yielding: {escaped + pre!r}, {field!r}, {spec!r}, {conv!r}')
						# field = eval(self._format(field), self._world)
						yield escaped + pre, field, spec, conv
						escaped = ''
					start_idx = -1
				idx = close_idx + 1



def pformat(s, **vars):
	"""
	Evaluates the keys in the given string as expressions using the given variables (recursively)
	"""
	fmt = PowerFormatter()
	return fmt.format(s, **vars)



def verbalize_list(terms, *, conjunction='and'):
	terms = list(terms)
	if len(terms) == 0:
		return ''
	if len(terms) == 1:
		return terms[0]
	if len(terms) == 2:
		return f'{terms[0]} {conjunction} {terms[1]}'
	return ', '.join(terms[:-1]) + f', {conjunction} {terms[-1]}'



class Parameterized(_Parameterized, fig.Configurable):
	def _extract_hparams(self, kwargs):
		params, remaining = super()._extract_hparams(kwargs)
		if self.my_config is not None:
			params.update({name: self.my_config.pull(name.replace('_', '-'), getattr(self, name, unspecified_argument))
			              for name, param in self.named_hyperparameters(hidden=True)
			              if name not in params and (param.default is not unspecified_argument or param.required)})
		return params, remaining



class Seeded(Parameterized):
	seed = hparam(None, inherit=True)

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._rng = np.random.default_rng(self.seed)



