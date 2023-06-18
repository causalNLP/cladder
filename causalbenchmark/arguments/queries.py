from .prompting import Prompter, ATE_Prompter, Mediation_Prompter
from .optim import RobustGapOptimization, ATEGapOptimization, PathGapOptimization


def create_query(key, *args, **kwargs):
	return _known_queries[key](*args, **kwargs)



_known_queries = {}
class Query(Prompter):
	def __init_subclass__(cls, **kwargs):
		super().__init_subclass__(**kwargs)
		name = getattr(cls, 'name', None)
		if name is not None:
			_known_queries[name] = cls



class ATE_Comparison(Query, ATE_Prompter, ATEGapOptimization):
	name = 'ate'



class Mediation_Comparison(Query, Mediation_Prompter, PathGapOptimization):
	name = 'med'


