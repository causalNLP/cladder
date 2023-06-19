from .imports import *
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



class ATE_Comparison(Query, ATE_Prompter):
	name = 'ate'
	
	_question_templates = {
		'is_better': 'Is {first["do(X=1)"]} {"better" if polarity else "worse"} than {second["do(X=1)"]}?',
		'lead_to': 'Will {first["do(X=1)"]} lead to {first["Y={polarity}"]} more than {second["do(X=1)"]}?',
		'achieve': 'To achieve {first["Y={polarity}"]}, should {first["Xsubject"]} {first["do(X=1)"]} rather than {second["do(X=1)"]}?',
	}
	_answer_templates = '{"Yes" if {answer} else "No"}.'
	@classmethod
	def verbalize_questions(cls, winner, loser, *, polarity=None, flip=None):
		polarities = [True, False] if polarity is None else [polarity]
		flips = [True, False] if flip is None else [flip]
		
		for polarity, flip in product(polarities, flips):
			first, second = (loser, winner) if flip else (winner, loser)
			
			answer = polarity ^ flip
			
			for tmpl, template in cls._question_templates.items():
				line = util.pformat(template, polarity=int(polarity), first=first, second=second)
				line = line[0].upper() + line[1:]
				ans = util.pformat(cls._answer_templates, answer=answer, polarity=int(polarity),
				                   first=first, second=second)
				yield {'template': tmpl, 'verb': line, 'verb_answer': ans,
				       'polarity': polarity, 'flip': flip, 'answer': answer}
				
			

class Mediation_Comparison(Query, Mediation_Prompter):
	name = 'med'


