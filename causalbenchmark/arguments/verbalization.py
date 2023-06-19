from .imports import *

from .stories import get_all_stories
from .commonsense import commonsense_score, beta_agreement_score, iou



def parse_term(term):
	var, *given = term.split('|')
	parents = None
	if len(given):
		parents = {k: int(v) for k,v in [x.split('=') for x in given[0].split(',')]}

	value = None
	if '=' in var:
		var, value = var.split('=')
		value = int(value)

	return var, value, parents



def set_term_value(term, value=None, parents=None):
	var, old_value, old_parents = parse_term(term)
	
	if parents is None:
		parents = old_parents
	parents = ','.join(f'{k}={v}' for k,v in parents.items()) if parents is not None else ''
	
	head = f'{var}={value}' if value is not None else var
	tail = f'|{parents}' if len(parents) else ''
	
	return head + tail
	
	

ambiguous_marginal_templates = {
	# # 'always': 'Almost always, {{var}={value}_sentence}.',
	# 'often': '{{var}={value}_sentence} often.',
	# 'more_often_than_not': '{{var}={value}_sentence} more often than not.',
	# 'half_the_time': '{{var}={value}_sentence} about half the time.',
	# # 'rarely': '{{var}={value}_sentence} rarely.',
	# # 'never': '{{var}={value}_sentence} almost never.',
	# 'more_often_than_not': '{{var}subject} {{var}={value}_verb} more often than not.',
	# 'half_the_time': '{{var}subject} {{var}={value}_verb} about half the time.',
	
	'often': '{{var}subject} often {{var}={value}_verb}.',
	'rarely': '{{var}subject} rarely {{var}={value}_verb}.',
	'usually': '{{var}subject} usually {{var}={value}_verb}.',
	'generally': '{{var}subject} generally {{var}={value}_verb}.',
	
	'very_likely': 'It is very likely that {{var}subject} {{var}={value}_verb}.',
	'likely': 'It is likely that {{var}subject} {{var}={value}_verb}.',
	'somewhat_likely': 'It is somewhat likely that {{var}subject} {{var}={value}_verb}.',
	'unlikely': 'It is unlikely that {{var}subject} {{var}={value}_verb}.',
	'very_unlikely': 'It is very unlikely that {{var}subject} {{var}={value}_verb}.',
	
	'very_few': 'Very few {{var}subject} {{var}={value}_verb}.',
	'few': 'Few {{var}subject} {{var}={value}_verb}.',
	'some': 'Some {{var}subject} {{var}={value}_verb}.',
	'many': 'Many {{var}subject} {{var}={value}_verb}.',
	'very_many': 'Very many {{var}subject} {{var}={value}_verb}.',
	
	# 'very_high': 'The probability that {{var}subject} {{var}={value}_verb} is very high.',
	'high': 'The probability that {{var}subject} {{var}={value}_verb} is high.',
	# 'moderate': 'The probability that {{var}subject} {{var}={value}_verb} is moderate.',
	'significant': 'The probability that {{var}subject} {{var}={value}_verb} is significant.',
	'low': 'The probability that {{var}subject} {{var}={value}_verb} is low.',
	# 'very_low': 'The probability that {{var}subject} {{var}={value}_verb} is very low.',
	
	# 'certain': 'It is certain that {{var}subject} {{var}={value}_verb}.',
	'almost_certain': 'It is almost certain that {{var}subject} {{var}={value}_verb}.',
	'probable': 'It is probable that {{var}subject} {{var}={value}_verb}.',
	# 'possible': 'It is possible that {{var}subject} {{var}={value}_verb}.',
	'improbable': 'It is improbable that {{var}subject} {{var}={value}_verb}.',
	'almost_impossible': 'It is almost impossible that {{var}subject} {{var}={value}_verb}.',
	# 'impossible': 'It is impossible that {{var}subject} {{var}={value}_verb}.',
	
}


ambiguous_marginal_default_implications = {
	'often': [0.7, 1],
	'rarely': [0, 0.3],
	'usually': [0.6, 1],
	'generally': [0.4, 1],
	# 'more_often_than_not': [0.5, 1],
	# 'half_the_time': [0.3, 0.7],
	
	'very_likely': [0.8, 1],
	'likely': [0.7, 0.9],
	'somewhat_likely': [0.4, 0.7],
	'unlikely': [0.1, 0.3],
	'very_unlikely': [0, 0.2],
	
	'very_few': [0, 0.2],
	'few': [0.1, 0.3],
	'some': [0.3, 0.7],
	'many': [0.7, 0.9],
	'very_many': [0.8, 1],
	
	# 'very_high': [0.8, 1],
	'high': [0.6, 0.8],
	# 'moderate': [0.4, 0.6],
	'significant': [0.3, 0.9],
	'low': [0.2, 0.4],
	# 'very_low': [0, 0.2],
	
	'almost_certain': [0.9, 1],
	'probable': [0.6, 0.9],
	'improbable': [0.1, 0.4],
}

ambiguous_conditional_templates = {
	'often': '{{var}subject} often {{var}={value}_end} {head}.',
	'rarely': '{{var}subject} rarely {{var}={value}_end} {head}.',
	'usually': '{{var}subject} usually {{var}={value}_end} {head}.',
	'generally': '{{var}subject} generally {{var}={value}_end} {head}.',
	# 'more_often_than_not': '{{var}subject} {{var}={value}_end} more often than not {head}.',
	# 'half_the_time': '{{var}subject} {{var}={value}_end} about half the time {head}.',
	
	'very_likely': '{head}, it is very likely that {{var}subject} {{var}={value}_end}.',
	'likely': '{head}, it is likely that {{var}subject} {{var}={value}_end}.',
	'somewhat_likely': '{head}, it is somewhat likely that {{var}subject} {{var}={value}_end}.',
	'unlikely': '{head}, it is unlikely that {{var}subject} {{var}={value}_end}.',
	'very_unlikely': '{head}, it is very unlikely that {{var}subject} {{var}={value}_end}.',
	
	'very_few': '{head}, very few {{var}subject} {{var}={value}_end}.',
	'few': '{head}, few {{var}subject} {{var}={value}_end}.',
	'some': '{head}, some {{var}subject} {{var}={value}_end}.',
	'many': '{head}, many {{var}subject} {{var}={value}_end}.',
	'very_many': '{head}, very many {{var}subject} {{var}={value}_end}.',
	
	'high': '{head}, the probability that {{var}subject} {{var}={value}_end} is high.',
	# 'moderate': '{head}, the probability that {{var}subject} {{var}={value}_end} is moderate.',
	'significant': '{head}, the probability that {{var}subject} {{var}={value}_end} is significant.',
	'low': '{head}, the probability that {{var}subject} {{var}={value}_end} is low.',
	
	'almost_certain': '{head}, it is almost certain that {{var}subject} {{var}={value}_end}.',
	'probable': '{head}, it is probable that {{var}subject} {{var}={value}_end}.',
	'improbable': '{head}, it is improbable that {{var}subject} {{var}={value}_end}.',
	'almost_impossible': '{head}, it is almost impossible that {{var}subject} {{var}={value}_end}.',
	
}

default_conditional_heads = {
	'when': 'when {{var}subject} {{var}={value}_verb}',
	# 'for': 'for {{var}subject} that {{var}={value}_verb}',
	# 'if': 'if {conditions}, ',
}

ambiguous_conditional_default_implications = {
	'often': [0.7, 1],
	'rarely': [0, 0.3],
	'usually': [0.6, 1],
	'generally': [0.4, 1],
	# 'more_often_than_not': [0.5, 1],
	# 'half_the_time': [0.3, 0.7],

	'very_likely': [0.8, 1],
	'likely': [0.7, 0.9],
	'somewhat_likely': [0.4, 0.7],
	'unlikely': [0.1, 0.3],
	'very_unlikely': [0, 0.2],

	'very_few': [0, 0.2],
	'few': [0.1, 0.3],
	'some': [0.3, 0.7],
	'many': [0.7, 0.9],
	'very_many': [0.8, 1],
	
	# 'very_high': [0.8, 1],
	'high': [0.6, 0.8],
	# 'moderate': [0.4, 0.6],
	'significant': [0.3, 0.9],
	'low': [0.2, 0.4],
	# 'very_low': [0, 0.2],

	'almost_certain': [0.9, 1],
	'probable': [0.6, 0.9],
	'improbable': [0.1, 0.4],
	'almost_impossible': [0, 0.1],
}


default_sequence_template = '{{var}={value}_verb}'


def verbalize_conditions(story, parents):
	conds = []
	
	parent_list = list(parents.items())
	custom_heads = {v: story[f'{v}={val}_head'] for v, val in parents.items() if f'{v}={val}_head' in story}
	assert len(custom_heads) <= 1, 'Only one custom head is allowed'
	
	# sort parents by custom head
	if len(custom_heads):
		parent_list = sorted(parent_list, key=lambda x: x[0] not in custom_heads)
	
	subject_head_index = None
	for i, (v, val) in enumerate(parent_list):
		if v in custom_heads:
			conds.append(custom_heads[v])
		else:
			if subject_head_index is None:
				subject_head_index = i
				conds.append(None)  # placeholder for subject head
			else:
				conds.append(util.pformat(default_sequence_template, var=v, value=val, **story))
	
	for head_key in ([None] if subject_head_index is None else default_conditional_heads):
		if head_key is not None:
			conds[subject_head_index] = util.pformat(default_conditional_heads[head_key],
			                                         var=parent_list[subject_head_index][0],
			                                         value=parent_list[subject_head_index][1],
			                                         **story)
		
		head = util.verbalize_list(conds) \
			if len(custom_heads) == 0 or len(conds) - len(custom_heads) > 1 \
			else ', '.join(conds)
		yield head, head_key, parent_list


def verbalize_ambiguous_evidence(story, term, evidence_table=None, avoid_double_negative=True):
	
	var, value, parents = parse_term(term)
	if value is None:
		value = 1
	
	if evidence_table is None:
		evidence_table = ambiguous_marginal_default_implications if parents is None \
			else ambiguous_conditional_default_implications
	evidence_table.update(story.get('evidence_table', {}).get(set_term_value(term), {}))
	
	if value == 0:
		evidence_table = {tmpl: [1 - ub, 1 - lb] for tmpl, (lb, ub) in evidence_table.items()
		         if not avoid_double_negative or ((lb+ub)/2 > 0.5)}
	
	if parents is None:
		for tmpl, bounds in evidence_table.items():
			if tmpl in ambiguous_marginal_templates:
				lb, ub = bounds
				mean = (lb + ub) / 2
				
				template = ambiguous_marginal_templates[tmpl]
				line = util.pformat(template, var=var, value=value, mean=mean, lb=lb, ub=ub, **story)
				line = line[0].upper() + line[1:]
				yield {'key': tmpl, 'implication': bounds, 'verb': line}
		
	else:
		for head, head_key, parent_list in verbalize_conditions(story, parents):
			for tmpl, bounds in evidence_table.items():
				if tmpl in ambiguous_conditional_templates:
					lb, ub = bounds
					mean = (lb + ub) / 2
					template = ambiguous_conditional_templates[tmpl]
					line = util.pformat(template, var=var, value=value, mean=mean, lb=lb, ub=ub,
					                    head=head, parents=parent_list,
					                    **story)
					line = line[0].upper() + line[1:]
					
					out = {'key': tmpl, 'implication': bounds, 'verb': line}# 'given': parents}
					if head_key is not None:
						out['head'] = head_key
					yield out
	


def generate_ambiguous_evidence(story, term, *, evidence_table=None, avoid_double_negative=True,
                                allow_flips=True, fill_missing=True):
	'''
	
	:param story:
	:param term:
	:param allow_flips: flip value of term (e.g. X=0 -> X=1)
	:param fill_missing: supplement missing terms with default implications
	:return:
	'''
	
	var, value, parents = parse_term(term)
	if value is None:
		value = 1
	
	if evidence_table is not None and fill_missing:
		evidence_table = evidence_table.copy()
		evidence_table.update(ambiguous_marginal_default_implications if parents is None
		                      else ambiguous_conditional_default_implications)
	
	yield from verbalize_ambiguous_evidence(story, term, evidence_table=evidence_table,
	                                        avoid_double_negative=avoid_double_negative)
	
	if allow_flips:
		yield from verbalize_ambiguous_evidence(story, set_term_value(term, 1-value), evidence_table=evidence_table,
		                                        avoid_double_negative=avoid_double_negative)
	


def test_verbalize_ambiguous_evidence():
	# random.seed(0)
	gen = np.random.RandomState(1)
	stories = get_all_stories()
	story = random.choice(random.choice(stories))
	# story = stories[0][1]
	
	terms = list(story['commonsense'])
	
	name = story['name']
	
	# term = 'Y=0|X=1,U=1'
	# possible = list(generate_ambiguous_evidence(story, term))
	# print(tabulate(possible, headers='keys'))
	print()
	
	commonsense = story['commonsense']
	
	border = '-'*50
	for term in terms:
		possible = list(generate_ambiguous_evidence(story, term))
		
		for item in possible:
			# item['score'] = beta_agreement_score(*commonsense[term], *item['implication'])
			item['score'] = iou(*commonsense[term], *item['implication'])
		
		picks = gen.choice(possible, size=10, replace=False) if len(possible) > 10 else possible
		
		picks = [{'verb': item['verb'], 'score': item['score']*100} for item in picks]
		
		print(border)
		print(term)
		print(sorted([int(100*item["score"]) for item in possible], reverse=True))
		print(tabulate(picks, headers='keys'))
		# print(border)
		
	print(border)



ambiguous_marginal_value_templates = {
	'about': 'There is about a {mean:.0%} chance that {{var}subject} {{var}={value}_verb}.',
	'around': 'There is around a {mean:.0%} chance that {{var}subject} {{var}={value}_verb}.',
	
	'roughly': 'Roughly {mean:.0%} of {{var}subject} {{var}={value}_verb}.',
	'approximately': 'Approximately, {mean:.0%} of {{var}subject} {{var}={value}_verb}.',
	
	'give_or_take': 'The chance that {{var}subject} {{var}={value}_verb} is {mean:.0%}, give or take.',
	'more_or_less': 'The chance that {{var}subject} {{var}={value}_verb} is more or less {mean:.0%}.',
}


ambiguous_marginal_value_implications = {
	'about': [-0.05, 0.05],
	'around': [-0.1, 0.1],
	
	'approximately': [-0.05, 0.05],
	'roughly': [-0.1, 0.1],
	
	'give_or_take': [-0.05, 0.05],
	'more_or_less': [-0.1, 0.1],
}


ambiguous_conditional_value_templates = {
	'about': '{head}, there is about a {mean:.0%} chance that {{var}subject} {{var}={value}_end}.',
	'around': '{head}, there is around a {mean:.0%} chance that {{var}subject} {{var}={value}_end}.',
	
	'roughly': '{head}, roughly {mean:.0%} of {{var}subject} {{var}={value}_end}.',
	'approximately': '{head}, approximately {mean:.0%} of {{var}subject} {{var}={value}_end}.',
	
	'give_or_take': '{head}, the chance that {{var}subject} {{var}={value}_end} is {mean:.0%}, give or take.',
	'more_or_less': '{head}, the chance that {{var}subject} {{var}={value}_end} is more or less {mean:.0%}.',
}


ambiguous_conditional_value_implications = {
	'about': [-0.05, 0.05],
	'around': [-0.1, 0.1],
	
	'approximately': [-0.05, 0.05],
	'roughly': [-0.1, 0.1],
	
	'give_or_take': [-0.05, 0.05],
	'more_or_less': [-0.1, 0.1],
}



def verbalize_ambiguous_evidence_from_value(story, term, mean, *, evidence_table=None, avoid_double_negative=True):
	
	var, value, parents = parse_term(term)
	if value is None:
		value = 1

	if evidence_table is None:
		evidence_table = ambiguous_marginal_value_implications.copy() if parents is None \
			else ambiguous_conditional_value_implications.copy()
	evidence_table.update(story.get('evidence_table', {}).get(set_term_value(term), {}))
	
	if value == 0:
		evidence_table = {tmpl: [1 - ub, 1 - lb] for tmpl, (lb, ub) in evidence_table.items()
		                  if not avoid_double_negative or ((lb + ub) / 2 > 0.5)}
	
	if parents is None:
		
		for tmpl, bounds in evidence_table.items():
			if tmpl not in ambiguous_marginal_value_templates:
				continue
			lb, ub = bounds
			lb = max(mean + lb, 0.)
			ub = min(mean + ub, 1.)
			
			template = ambiguous_marginal_value_templates[tmpl]
			
			line = util.pformat(template, var=var, value=value, mean=mean, lb=lb, ub=ub, **story)
			line = line[0].upper() + line[1:]
			yield {'key': tmpl, 'implication': [lb, ub], 'verb': line}
		
	else:
		
		for head, head_key, parent_list in verbalize_conditions(story, parents):
			for tmpl, bounds in evidence_table.items():
				if tmpl in ambiguous_conditional_value_templates:
					lb, ub = bounds
					lb = max(mean + lb, 0.)
					ub = min(mean + ub, 1.)
					
					template = ambiguous_conditional_value_templates[tmpl]
					line = util.pformat(template, var=var, value=value, mean=mean, lb=lb, ub=ub,
					                    head=head, parents=parent_list,
					                    **story)
					line = line[0].upper() + line[1:]
					
					out = {'key': tmpl, 'implication': bounds, 'verb': line}  # 'given': parents}
					if head_key is not None:
						out['head'] = head_key
					yield out


def test_verbalize_ambiguous_value_evidence():
	gen = np.random.RandomState(1)
	stories = get_all_stories()
	story = random.choice(random.choice(stories))
	# story = stories[0][1]
	
	terms = list(story['commonsense'])
	
	print()

	name = story['name']
	print(name)
	
	commonsense = story['commonsense']
	
	border = '-' * 50
	for term in terms:
		possible = list(verbalize_ambiguous_evidence_from_value(story, term, gen.random()))
		
		for item in possible:
			# item['score'] = beta_agreement_score(*commonsense[term], *item['implication'])
			item['score'] = iou(*commonsense[term], *item['implication'])
		
		picks = gen.choice(possible, size=10, replace=False) if len(possible) > 10 else possible
		
		picks = [{'verb': item['verb'], 'score': item['score'] * 100} for item in picks]
		
		print(border)
		print(term)
		print(sorted([int(100 * item["score"]) for item in possible], reverse=True))
		print(tabulate(picks, headers='keys'))
	
	print(border)


precise_marginal_templates = {
	'at_least': 'There is at least a {lb:.0%} chance that {{var}subject} {{var}={value}_verb}.',
	'at_most': 'There is at most a {ub:.0%} chance that {{var}subject} {{var}={value}_verb}.',
	
	'least': 'At least {lb:.0%} of {{var}subject} {{var}={value}_verb}.',
	'most': 'At most {ub:.0%} of {{var}subject} {{var}={value}_verb}.',
	
	'with': 'With a probability of {lb*100:.0f} to {ub:.0%}, {{var}subject} {{var}={value}_verb}.',
	'range': 'There is a {lb*100:.0f}-{ub:.0%} chance that {{var}subject} {{var}={value}_verb}.',
	'between': 'Between {lb*100:.0f} and {ub:.0%} of the time {{var}subject} {{var}={value}_verb}.',
	'short': '{lb*100:.0f} to {ub:.0%} of {{var}subject} {{var}={value}_verb}.',
}


precise_marginal_implications = {
	'at_least': [None, .99],
	'at_most': [0.01, None],
	
	'least': [None, .99],
	'most': [0.01, None],
	
	'with': [None, None],
	'range': [None, None],
	'between': [None, None],
	'short': [None, None],
}


precise_conditional_templates = {
	'at_least': '{head}, there is at least a {lb:.0%} chance that {{var}subject} {{var}={value}_verb}.',
	'at_most': '{head}, there is at most a {ub:.0%} chance that {{var}subject} {{var}={value}_verb}.',
	
	'least': '{head}, at least {lb:.0%} of {{var}subject} {{var}={value}_verb}.',
	'most': '{head}, at most {ub:.0%} of {{var}subject} {{var}={value}_verb}.',
	
	'with': '{head}, with a probability of {lb*100:.0f}-{ub:.0%}, {{var}subject} {{var}={value}_verb}.',
	'range': '{head}, there is a {lb*100:.0f}-{ub:.0%} chance that {{var}subject} {{var}={value}_verb}.',
	'between': '{head}, between {lb:.0%} and {ub:.0%} of the time {{var}subject} {{var}={value}_verb}.',
	'short': '{head}, {lb*100:.0f} to {ub:.0%} of {{var}subject} {{var}={value}_verb}.',
}


precise_conditional_implications = {
	'at_least': [None, .99],
	'at_most': [0.01, None],
	
	'least': [None, .99],
	'most': [0.01, None],
	
	'with': [None, None],
	'range': [None, None],
	'between': [None, None],
	'short': [None, None],
}


def verbalize_precise_evidence(story, term, bounds, *, evidence_table=None):
	lb, ub = bounds
	mean = (lb + ub) / 2
	
	var, value, parents = parse_term(term)
	if value is None:
		value = 1
	
	if evidence_table is None:
		evidence_table = precise_marginal_implications.copy() if parents is None \
			else precise_conditional_implications.copy()
	evidence_table.update(story.get('evidence_table', {}).get(set_term_value(term), {}))
	
	if parents is None:
		
		for tmpl, lims in evidence_table.items():
			ll, ul = lims
			if tmpl in precise_marginal_templates and (ll is None or ll >= lb) and (ul is None or ul <= ub):
				template = precise_marginal_templates[tmpl]
				line = util.pformat(template, var=var, value=value, mean=mean, lb=lb, ub=ub, **story)
				line = line[0].upper() + line[1:]
				yield {'key': tmpl, 'implication': bounds, 'verb': line}
	
	else:
		
		for head, head_key, parent_list in verbalize_conditions(story, parents):
			for tmpl, lims in evidence_table.items():
				ll, ul = lims
				if tmpl in precise_conditional_templates and (ll is None or ll >= lb) and (ul is None or ul <= ub):
					template = precise_conditional_templates[tmpl]
					line = util.pformat(template, var=var, value=value, mean=mean, lb=lb, ub=ub,
					                    head=head, parents=parent_list,
					                    **story)
					line = line[0].upper() + line[1:]
					
					out = {'key': tmpl, 'implication': bounds, 'verb': line}
					if head_key is not None:
						out['head'] = head_key
					yield out
	


def test_verbalize_precise_evidence():
	gen = np.random.RandomState(1)
	stories = get_all_stories()
	story = random.choice(random.choice(stories))
	
	terms = list(story['commonsense'])
	
	print()
	
	name = story['name']
	print(name)
	
	commonsense = story['commonsense']
	
	border = '-' * 50
	for term in terms:
		possible = list(verbalize_precise_evidence(story, term, sorted([gen.random(), gen.random()])))
		
		for item in possible:
			# item['score'] = beta_agreement_score(*commonsense[term], *item['implication'])
			item['score'] = iou(*commonsense[term], *item['implication'])
		
		picks = gen.choice(possible, size=10, replace=False) if len(possible) > 10 else possible
		
		picks = [{'verb': item['verb'], 'score': item['score'] * 100} for item in picks]
		
		print(border)
		print(term)
		print(sorted([int(100 * item["score"]) for item in possible], reverse=True))
		print(tabulate(picks, headers='keys'))
	
	print(border)





