from omnibelt import Class_Registry
import omnifig as fig
from ..util import Parameterized, hparam



_query_registry = Class_Registry()
_register_query = _query_registry.get_decorator()

get_query_class = _query_registry.get_class


class register_query(_register_query):
	def __init__(self, name, *args, **kwargs):
		super().__init__(name, *args, **kwargs)
		self._name = name


	def __call__(self, cls):
		if getattr(cls, '_query_name', None) is None:
			setattr(cls, '_query_name', self._name)
		fig.component(self._name)(cls)
		return super().__call__(cls)



class QueryFailedError(Exception):
	pass



def create_query(query_name, *args, **kwargs):
	return _query_registry.get_class(query_name)(*args, **kwargs)



class AbstractQueryType(Parameterized):
	'''Automatically registers subclasses with the query_registry (if a name is provided).'''
	def __init_subclass__(cls, name=None, **kwargs):
		super().__init_subclass__(**kwargs)
		cls._query_name = name
		if name is not None:
			register_query(name)(cls)
			fig.component(name)(cls)


	def __str__(self):
		return self._query_name


	@property
	def query_name(self):
		return self._query_name


	_QueryFailedError = QueryFailedError

	
	# @classmethod
	# def number_of_questions(cls, scm):
	# 	'''computes the number of questions for a certain query type with the given SCM'''
	# 	raise NotImplementedError


	def meta_data(self, scm, labels):
		raise NotImplementedError


	def reasoning(self, scm, labels, entry, **details):
		raise NotImplementedError


	def symbolic_given_info(self, scm, labels):
		raise NotImplementedError


	def verbalize_reasoning(self, scm, labels, **details):
		raise NotImplementedError


	def verbalize_given_info(self, scm, labels, **details):
		'''verbalizes the given information necessary to solve a query'''
		raise NotImplementedError


	def verbalize_background(self, scm, labels, **details):
		'''verbalizes the background information necessary to solve a query'''
		return scm.verbalize_background(labels)


	def generate_questions(self, scm, labels):
		'''lazily generates all question/answer pairs for a certain query type with the given SCM and labels'''
		yield from ()



class MetricQueryType(AbstractQueryType):
	answer_template = hparam('{{{-1:"no", 0:"unknown", 1:"yes"}}[int({answer})]}', inherit=True)

	treatment = hparam('X', inherit=True)
	outcome = hparam('Y', inherit=True)



# class PhenomenonQueryType(AbstractQueryType):
# 	_sol_verbalization = {True:'yes', False:'no', None: 'not enough information'}
#
# 	def collect_background(self, scm, labels):
# 		background = scm.description(labels)
# 		details = scm.details(labels)
# 		if details is not None:
# 			background += ' ' + details
# 		return background




