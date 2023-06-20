from .imports import *

from .stories import get_all_stories
from .commonsense import commonsense_score, beta_agreement_score, iou, simple_containment



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
	'often': '{{var}subject} often {{var}={value}_end}',
	'rarely': '{{var}subject} rarely {{var}={value}_end}',
	'usually': '{{var}subject} usually {{var}={value}_end}',
	'generally': '{{var}subject} generally {{var}={value}_end}',
	# 'more_often_than_not': '{{var}subject} {{var}={value}_end} more often than not',
	# 'half_the_time': '{{var}subject} {{var}={value}_end} about half the time',
	
	'very_likely': 'it is very likely that {{var}subject} {{var}={value}_end}',
	'likely': 'it is likely that {{var}subject} {{var}={value}_end}',
	'somewhat_likely': 'it is somewhat likely that {{var}subject} {{var}={value}_end}',
	'unlikely': 'it is unlikely that {{var}subject} {{var}={value}_end}',
	'very_unlikely': 'it is very unlikely that {{var}subject} {{var}={value}_end}',
	
	'very_few': 'very few {{var}subject} {{var}={value}_end}',
	'few': 'few {{var}subject} {{var}={value}_end}',
	'some': 'some {{var}subject} {{var}={value}_end}',
	'many': 'many {{var}subject} {{var}={value}_end}',
	'very_many': 'very many {{var}subject} {{var}={value}_end}',

	'very_high': 'the probability that {{var}subject} {{var}={value}_end} is very high',
	'high': 'the probability that {{var}subject} {{var}={value}_end} is high',
	# 'moderate': 'the probability that {{var}subject} {{var}={value}_end} is moderate',
	'significant': 'the probability that {{var}subject} {{var}={value}_end} is significant',
	'low': 'the probability that {{var}subject} {{var}={value}_end} is low',
	'very_low': 'the probability that {{var}subject} {{var}={value}_end} is very low',
	
	'almost_certain': 'it is almost certain that {{var}subject} {{var}={value}_end}',
	'probable': 'it is probable that {{var}subject} {{var}={value}_end}',
	'improbable': 'it is improbable that {{var}subject} {{var}={value}_end}',
	'almost_impossible': 'it is almost impossible that {{var}subject} {{var}={value}_end}',
	
}

default_conditional_heads = {
	'when': 'when {{var}subject} {{var}={value}_verb}',
	# 'for': 'for {{var}subject} that {{var}={value}_verb}',
	'if': 'if {{var}subject} {{var}={value}_verb}',
}

default_conditional_structure = {
	'first': '{head}, {tail}.',
	'last': '{tail} {head}.',
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
	
	'very_high': [0.8, 1],
	'high': [0.6, 0.8],
	# 'moderate': [0.4, 0.6],
	'significant': [0.3, 0.9],
	'low': [0.2, 0.4],
	'very_low': [0, 0.2],

	'almost_certain': [0.9, 1],
	'probable': [0.6, 0.9],
	'improbable': [0.1, 0.4],
	'almost_impossible': [0, 0.1],
}


default_sequence_template = '{{var}={value}_verb}'


def _verbalize_conditions(story, parents, *, gen=None):
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

	head_options = [None] if subject_head_index is None else list(default_conditional_heads.keys())
	if gen is not None:
		gen.shuffle(head_options)
	for head_key in head_options:
		if head_key is not None:
			conds[subject_head_index] = util.pformat(default_conditional_heads[head_key],
			                                         var=parent_list[subject_head_index][0],
			                                         value=parent_list[subject_head_index][1],
			                                         **story)
		
		head = util.verbalize_list(conds) \
			if len(custom_heads) == 0 or len(conds) - len(custom_heads) > 1 \
			else ', '.join(conds)
		yield head, head_key, parent_list


def verbalize_ambiguous_evidence(story, term, evidence_table=None, *, avoid_double_negative=True, prior=None,
                                 prior_agreement=0.5, gen=None):
	
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

	evidence_order = list(evidence_table.keys())
	if gen is not None:
		gen.shuffle(evidence_order)

	headers = [[None, None, None]] if parents is None else _verbalize_conditions(story, parents, gen=gen)
	templates = ambiguous_marginal_templates if parents is None else ambiguous_conditional_templates
	structures = [[None, None]] if parents is None else default_conditional_structure.items()

	for head, head_key, parent_list in headers:
		for strc, code in structures:
			for tmpl in evidence_order:
				if tmpl in templates:
					bounds = evidence_table[tmpl]
					lb, ub = bounds
					if prior is not None and simple_containment(*prior, lb, ub) < prior_agreement:
						continue

					mean = (lb + ub) / 2
					template = templates[tmpl]
					line = util.pformat(template, var=var, value=value, mean=mean, lb=lb, ub=ub,
					                    parents=parent_list,
					                    **story)
					if strc is not None:
						line = util.pformat(code, tail=line, head=head, **story)
					line = line[0].upper() + line[1:]

					out = {'key': tmpl, 'implication': bounds, 'verb': line, 'ID': tmpl, 'type': 'ambiguous'}# 'given': parents}
					if head_key is not None:
						out['head'] = head_key
						out['clause'] = strc
						out['ID'] += f'-{head_key}-{strc}'
					yield out



def generate_ambiguous_evidence(story, term, prior=None, *, evidence_table=None, avoid_double_negative=True,
                                allow_flips=True, agreement_threshold=None, gen=None):
	'''
	
	:param story:
	:param term:
	:param prior: not needed here because all ambiguous evidencee use predefined ranges
	:param allow_flips: flip value of term (e.g. X=0 -> X=1)
	:param fill_missing: supplement missing terms with default implications
	:return:
	'''
	
	var, value, parents = parse_term(term)
	if value is None:
		value = 1

	yield from verbalize_ambiguous_evidence(story, term, evidence_table=evidence_table, prior=prior,
	                                        avoid_double_negative=avoid_double_negative, gen=gen)
	
	if allow_flips:
		# if prior is not None:
		# 	prior = [1 - prior[1], 1 - prior[0]]
		for item in verbalize_ambiguous_evidence(story, set_term_value(term, 1-value), evidence_table=evidence_table,
		                                        avoid_double_negative=avoid_double_negative, gen=gen, prior=prior):
			item['ID'] += '-flip'
			yield item
	


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
		possible = list(generate_ambiguous_evidence(story, term, gen=gen))
		
		for item in possible:
			# item['score'] = beta_agreement_score(*commonsense[term], *item['implication'])
			item['score'] = iou(*commonsense[term], *item['implication'])

		picks = [{'verb': item['verb'], 'score': item['score']*100} for item in possible[:10]]
		
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

	# 'close_to': [-0.02, 0.02],
	# 'near': [-0.05, 0.05],
	# 'nearly': [-0.1, 0.1],

	'approximately': [-0.05, 0.05],
	'roughly': [-0.1, 0.1],
	
	'give_or_take': [-0.05, 0.05],
	'more_or_less': [-0.1, 0.1],
}


ambiguous_conditional_value_templates = {
	'about': 'there is about a {mean:.0%} chance that {{var}subject} {{var}={value}_end}',
	'around': 'there is around a {mean:.0%} chance that {{var}subject} {{var}={value}_end}',
	
	'roughly': 'roughly {mean:.0%} of {{var}subject} {{var}={value}_end}',
	'approximately': 'approximately {mean:.0%} of {{var}subject} {{var}={value}_end}',
	
	'give_or_take': 'the chance that {{var}subject} {{var}={value}_end} is {mean:.0%}, give or take',
	'more_or_less': 'the chance that {{var}subject} {{var}={value}_end} is more or less {mean:.0%}',
}


ambiguous_conditional_value_implications = {
	'about': [-0.05, 0.05],
	'around': [-0.1, 0.1],
	
	'approximately': [-0.05, 0.05],
	'roughly': [-0.1, 0.1],
	
	'give_or_take': [-0.05, 0.05],
	'more_or_less': [-0.1, 0.1],
}



def verbalize_ambiguous_evidence_from_value(story, term, prob=None, *, evidence_table=None, avoid_double_negative=True,
                                            gen=None, agreement_threshold=None, prior=None):
	
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

	evidence_order = list(evidence_table.keys())
	if gen is not None:
		gen.shuffle(evidence_order)

	mean = prob

	headers = [[None, None, None]] if parents is None else _verbalize_conditions(story, parents, gen=gen)
	templates = ambiguous_marginal_value_templates if parents is None else ambiguous_conditional_value_templates
	structures = [[None, None]] if parents is None else default_conditional_structure.items()

	for head, head_key, parent_list in headers:
		for strc, code in structures:
			for tmpl in evidence_order:
				if tmpl in templates:
					bounds = evidence_table[tmpl]
					lb, ub = bounds
					if prob is None:
						assert prior is not None, 'Must provide either prob or prior'
						assert gen is not None, 'Must provide gen if prior is provided'
						lp, up = prior
						if agreement_threshold is not None:
							lp -= agreement_threshold * lb
							up -= agreement_threshold * ub
						mean = gen.uniform(lp, up) if lp < up else (lp + up) / 2
					lb = max(mean + lb, 0.)
					ub = min(mean + ub, 1.)

					template = templates[tmpl]
					line = util.pformat(template, var=var, value=value, mean=mean, lb=lb, ub=ub,
					                    head=head, parents=parent_list,
					                    **story)
					if strc is not None:
						line = util.pformat(code, tail=line, head=head, **story)
					line = line[0].upper() + line[1:]

					out = {'key': tmpl, 'implication': [lb, ub], 'verb': line, 'ID': tmpl, 'type': 'ambiguous'}
					if head_key is not None:
						out['head'] = head_key
						out['clause'] = strc
						out['ID'] += f'-{head_key}-{strc}'
					yield out



def generate_ambiguous_evidence_from_value(story, term, prior=None, *, evidence_table=None, avoid_double_negative=True,
                                allow_flips=True, agreement_threshold=None, gen=None):

	var, value, parents = parse_term(term)
	if value is None:
		value = 1

	yield from verbalize_ambiguous_evidence_from_value(story, term, evidence_table=evidence_table, prior=prior,
	                                        avoid_double_negative=avoid_double_negative, gen=gen,
	                                                   agreement_threshold=agreement_threshold)

	if allow_flips:
		if prior is not None:
			prior = [1 - prior[1], 1 - prior[0]]
		for item in verbalize_ambiguous_evidence_from_value(story, set_term_value(term, 1 - value),
		                                                   evidence_table=evidence_table,
		                                                   agreement_threshold=agreement_threshold,
		                                        avoid_double_negative=avoid_double_negative, gen=gen, prior=prior):
			item['ID'] += '-flip'
			yield item



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


interval_marginal_templates = {
	'at_least': 'There is at least a {lb:.0%} chance that {{var}subject} {{var}={value}_verb}.',
	'at_most': 'There is at most a {ub:.0%} chance that {{var}subject} {{var}={value}_verb}.',

	'up_to': 'Up to {ub:.0%} of {{var}subject} {{var}={value}_verb}.',

	'no_less_than': 'No less than {lb:.0%} of {{var}subject} {{var}={value}_verb}.',
	'no_more_than': 'No more than {ub:.0%} of {{var}subject} {{var}={value}_verb}.',
	
	'least': 'At least {lb:.0%} of {{var}subject} {{var}={value}_verb}.',
	'most': 'At most {ub:.0%} of {{var}subject} {{var}={value}_verb}.',
	
	'with': 'With a probability of {lb*100:.0f} to {ub:.0%}, {{var}subject} {{var}={value}_verb}.',
	'range': 'There is a {lb*100:.0f}-{ub:.0%} chance that {{var}subject} {{var}={value}_verb}.',
	'between': 'Between {lb*100:.0f} and {ub:.0%} of the time {{var}subject} {{var}={value}_verb}.',
	'short': '{lb*100:.0f}-{ub:.0%} of {{var}subject} {{var}={value}_verb}.',
}


interval_marginal_implications = {
	'at_least': [None, .99],
	'at_most': [0.01, None],

	'up_to': [0.01, None],

	'no_less_than': [None, .99],
	'no_more_than': [0.01, None],
	
	'least': [None, .99],
	'most': [0.01, None],
	
	'with': [None, None],
	'range': [None, None],
	'between': [None, None],
	'short': [None, None],
}


interval_conditional_templates = {
	'at_least': 'there is at least a {lb:.0%} chance that {{var}subject} {{var}={value}_verb}',
	'at_most': 'there is at most a {ub:.0%} chance that {{var}subject} {{var}={value}_verb}',

	'up_to': 'up to {ub:.0%} of {{var}subject} {{var}={value}_verb}',

	'no_less_than': 'no less than {lb:.0%} of {{var}subject} {{var}={value}_verb}',
	'no_more_than': 'no more than {ub:.0%} of {{var}subject} {{var}={value}_verb}',
	
	'least': 'at least {lb:.0%} of {{var}subject} {{var}={value}_verb}',
	'most': 'at most {ub:.0%} of {{var}subject} {{var}={value}_verb}',
	
	'with': 'with a probability of {lb*100:.0f}-{ub:.0%}, {{var}subject} {{var}={value}_verb}',
	'range': 'there is a {lb*100:.0f}-{ub:.0%} chance that {{var}subject} {{var}={value}_verb}',
	'between': 'between {lb:.0%} and {ub:.0%} of the time {{var}subject} {{var}={value}_verb}',
	'short': '{lb*100:.0f}-{ub:.0%} of {{var}subject} {{var}={value}_verb}',
	'plus_minus': '{{var}subject} {{var}={value}_verb} {mean:.0%} of the time, '
	              'plus or minus {(ub-lb)/2:.0%} points',
}


interval_conditional_implications = {
	'at_least': [None, .99],
	'at_most': [0.01, None],

	'up_to': [0.01, None],

	'no_less_than': [None, .99],
	'no_more_than': [0.01, None],
	
	'least': [None, .99],
	'most': [0.01, None],
	
	'with': [None, None],
	'range': [None, None],
	'between': [None, None],
	'short': [None, None],
	'plus_minus': [None, None],
}


def verbalize_interval_evidence(story, term, bounds=None, *, evidence_table=None, gen=None,
                               agreement_threshold=None, prior=None, min_width=0.1):
	if bounds is not None:
		lb, ub = bounds
		mean = (lb + ub) / 2
	
	var, value, parents = parse_term(term)
	if value is None:
		value = 1
	
	if evidence_table is None:
		evidence_table = interval_marginal_implications.copy() if parents is None \
			else interval_conditional_implications.copy()
	evidence_table.update(story.get('evidence_table', {}).get(set_term_value(term), {}))

	evidence_order = list(evidence_table.keys())
	if gen is not None:
		gen.shuffle(evidence_order)

	headers = [[None, None, None]] if parents is None else _verbalize_conditions(story, parents, gen=gen)
	templates = interval_marginal_templates if parents is None else interval_conditional_templates
	structures = [[None, None]] if parents is None else default_conditional_structure.items()

	for head, head_key, parent_list in headers:
		for strc, code in structures:
			for tmpl in evidence_order:
				lims = evidence_table[tmpl]
				ll, ul = lims

				if bounds is None:
					assert prior is not None, 'Must provide either prob or prior'
					assert gen is not None, 'Must provide gen if prior is provided'
					lp, up = prior
					mid1, mid2 = None, None
					if agreement_threshold is None:
						fuel = 10
						while fuel > 0 and (mid1 is None or (up - lp) < min_width or (mid2 - mid1 < min_width)):
							mid1, mid2 = sorted(gen.uniform(lp, up, size=(2,)))
							fuel -= 1
						if fuel == 0:
							mid1 = lp
							mid2 = up
					else: # TODO: sample from all intervals s.t. IOU > agreement_threshold
						mid1 = lp
						mid2 = up
						# w = up - lp
						# mid1 = gen.uniform(lp, up)
						# mid2 = mid1 + agreement_threshold
					lb = mid1 if ll is None else gen.uniform(0., ll)
					ub = mid2 if ul is None else gen.uniform(ul, 1.)
					mean = (lb + ub) / 2

				if tmpl in templates and (ll is None or ll >= lb) and (ul is None or ul <= ub):
					template = templates[tmpl]
					line = util.pformat(template, var=var, value=value, mean=mean, lb=lb, ub=ub,
					                    head=head, parents=parent_list,
					                    **story)
					if strc is not None:
						line = util.pformat(code, tail=line, head=head, **story)
					line = line[0].upper() + line[1:]

					out = {'key': tmpl, 'implication': [lb, ub], 'verb': line, 'ID': tmpl, 'type': 'interval'}
					if head_key is not None:
						out['head'] = head_key
						out['clause'] = strc
						out['ID'] += f'-{head_key}-{strc}'
					yield out
	


def generate_interval_evidence(story, term, prior=None, *, evidence_table=None,
                                allow_flips=True, agreement_threshold=None, gen=None):

	var, value, parents = parse_term(term)
	if value is None:
		value = 1

	yield from verbalize_interval_evidence(story, term, evidence_table=evidence_table, prior=prior,
	                                      gen=gen,
	                                                   agreement_threshold=agreement_threshold)

	if allow_flips:
		if prior is not None:
			prior = [1 - prior[1], 1 - prior[0]]
		for item in verbalize_interval_evidence(story, set_term_value(term, 1 - value),
		                                                   evidence_table=evidence_table,
		                                                   agreement_threshold=agreement_threshold,
		                                      gen=gen, prior=prior):
			item['ID'] += '-flip'
			yield item



def test_verbalize_interval_evidence():
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
		possible = list(verbalize_interval_evidence(story, term, sorted([gen.random(), gen.random()])))
		
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
	'number': '{mean:.0%} of {{var}subject} {{var}={value}_verb}.',
	'chance': 'There is a {mean:.0%} chance that {{var}subject} {{var}={value}_verb}.',
	'probability': 'The probability that {{var}subject} {{var}={value}_verb} is {mean:.0%}.',
	'likelihood': 'The likelihood that {{var}subject} {{var}={value}_verb} is {mean:.0%}.',
	'assume_prob': 'Assume that {{var}subject} {{var}={value}_verb} with probability {mean:.0%}.',
	'given': 'Given that {mean:.0%} of {{var}subject} {{var}={value}_verb}.',
	'known': 'It is known that {mean:.0%} of {{var}subject} {{var}={value}_verb}.',
}

precise_conditional_templates = {
	'number': '{mean:.0%} of {{var}subject} {{var}={value}_verb}',
	'chance': 'there is a {mean:.0%} chance that {{var}subject} {{var}={value}_verb}',
	'probability': 'the probability that {{var}subject} {{var}={value}_verb} is {mean:.0%}',
	'likelihood': 'the likelihood that {{var}subject} {{var}={value}_verb} is {mean:.0%}',
	'assume_prob': 'Assume that {{var}subject} {{var}={value}_verb} with probability {mean:.0%}',
	'given': 'Given that {mean:.0%} of {{var}subject} {{var}={value}_verb}',
	'known': 'It is known that {mean:.0%} of {{var}subject} {{var}={value}_verb}',
}


def verbalize_precise_evidence(story, term, prob=None, *, templates=None, avoid_double_negative=True,
                                            gen=None, agreement_threshold=None, prior=None, width=0.002):

	var, value, parents = parse_term(term)
	if value is None:
		value = 1

	mean = prob

	headers = [[None, None, None]] if parents is None else _verbalize_conditions(story, parents, gen=gen)
	if templates is None:
		templates = precise_marginal_templates if parents is None else precise_conditional_templates
	template_order = list(templates.keys())
	if gen is not None:
		gen.shuffle(template_order)

	structures = [[None, None]] if parents is None else default_conditional_structure.items()

	for head, head_key, parent_list in headers:
		for strc, code in structures:
			for tmpl in template_order:
				if prob is None:
					assert prior is not None, 'Must provide either prob or prior'
					assert gen is not None, 'Must provide gen if prior is provided'
					lp, up = prior
					mean = gen.uniform(lp+width/2, up-width/2)
				lb = max(mean + width/2, 0.)
				ub = min(mean + width/2, 1.)

				template = templates[tmpl]
				line = util.pformat(template, var=var, value=value, mean=mean, lb=lb, ub=ub,
				                    head=head, parents=parent_list,
				                    **story)
				if strc is not None:
					line = util.pformat(code, tail=line, head=head, **story)
				line = line[0].upper() + line[1:]

				out = {'key': tmpl, 'implication': [lb, ub], 'verb': line, 'ID': tmpl, 'type': 'precise'}
				if head_key is not None:
					out['head'] = head_key
					out['clause'] = strc
					out['ID'] += f'-{head_key}-{strc}'
				yield out



def generate_precise_evidence(story, term, prior=None, *,
                                allow_flips=True, agreement_threshold=None, gen=None):

	var, value, parents = parse_term(term)
	if value is None:
		value = 1

	yield from verbalize_precise_evidence(story, term, prior=prior,
	                                        gen=gen, agreement_threshold=agreement_threshold)

	if allow_flips:
		if prior is not None:
			prior = [1 - prior[1], 1 - prior[0]]
		for item in verbalize_precise_evidence(story, set_term_value(term, 1 - value),
		                                                   agreement_threshold=agreement_threshold,
		                                        gen=gen, prior=prior):
			item['ID'] += '-flip'
			yield item




