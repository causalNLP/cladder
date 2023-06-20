from .imports import *

from .stories import get_available_stories, iterate_scenarios, get_story_system
from .queries import create_query
from .optim import ATEGapOptimization
from .commonsense import iou, simple_containment, beta_agreement_score
from .verbalization import generate_ambiguous_evidence, \
	verbalize_ambiguous_evidence_from_value, verbalize_precise_evidence, \
	generate_ambiguous_evidence_from_value, generate_interval_evidence, generate_precise_evidence


class Generator:
	def __init__(self, stories=None, history=None, setups=None, seed=None, agreement_threshold=0.9,
	             num_alternates=1, anticommonsense=False, skip_commonsense_scores=False):
		if history is None:
			history = {}
		if setups is None:
			setups = {}

		stories = stories or list(get_available_stories())

		if anticommonsense:
			raise NotImplementedError

		assert len(stories) > 0, "No stories available"
		self.stories = stories
		self.history = history
		self.setups = setups
		self.gen = np.random.RandomState(seed)
		self.num_alternates = num_alternates
		self.operator = create_query('ate')
		self.skip_commonsense_scores = skip_commonsense_scores
		self.agreement_threshold = agreement_threshold


	def prepare(self):
		full = [list(iterate_scenarios(story)) for story in self.stories]
		story_options = {}
		for scens in full:
			if scens[0]['query'] == 'ate':
				assert len(scens) == 2

				scens[0]['system'] = get_story_system(scens[0])
				scens[1]['system'] = get_story_system(scens[1])

				story_options[scens[0]['ID']] = scens[0], scens[1]
				story_options[scens[1]['ID']] = scens[1], scens[0]
		self.story_options = story_options
		return self


	def update_history(self, q):
		story_name = q['ID']
		if story_name not in self.history['stories']:
			self.history['stories'][story_name] = 0
		self.history['stories'][story_name] += 1

		for premise in chain(q['winner']['premises'], q['loser']['premises']):
			spec = premise['type']
			if spec not in self.history['specificity']:
				self.history['specificity'][spec] = 0
			self.history['specificity'][spec] += 1
			ID = premise['ID']
			if ID not in self.history['premises']:
				self.history['premises'][ID] = 0
			self.history['premises'][ID] += 1
			if ID not in self.history['alternates']:
				self.history['alternates'][ID] = 0
			self.history['alternates'][ID] += 1

		for premise in chain((p for g in q['winner']['alternates'].values() for ps in g.values() for p in ps),
		                     (p for g in q['loser']['alternates'].values() for ps in g.values() for p in ps)):
			ID = premise['ID']
			if ID not in self.history['alternates']:
				self.history['alternates'][ID] = 0
			self.history['alternates'][ID] += 1



	def to_anticommonsense(self, lb, ub, *, safety=0.1, eps=1e-5):
		'''
		picks the larger interval of the complement, and flips a coin if they are equal.
		safety is the factor by which the interval is shrunk to make space from the boundary (0=no shrinking, 1=max).
		'''
		if lb > 1-ub + eps or (lb + eps < 1-ub and self.gen.rand() < 0.5):
			la, ua = eps, lb * (1-safety)
		else:
			la, ua = 1 - (1 - ub) * safety, 1.-eps
		la = max(0., min(ua-eps, la))
		ua = min(max(ua, la+eps), 1.)
		return la, ua


	def select_story(self):
		winner, loser = self.balanced_choose(self.gen, self.story_options, self.history.setdefault('stories', {}))
		return winner, loser


	def select_fixed(self, winner, loser=None):
		system = winner['system']

		dofs = len(system.ate_dofs) # for each system

		fixed = None
		while fixed is None or tuple(fixed) in self.setups.setdefault(winner['ID'], set()):
			num = self.gen.randint(0, dofs-1)
			fixed = sorted(self.gen.choice(system.ate_dofs, size=num, replace=False))
		self.setups[winner['ID']].add(tuple(fixed))
		return fixed


	def spawn_premises(self, story, term, prior, strict=False):
		yield from self.generate_ambiguous_premise(story, term, prior, strict=strict)
		yield from self.generate_interval_premise(story, term, prior, strict=strict)
		# yield from self.generate_precise_premise(story, term, prior) # dont start with precise -> no where else to go for alternates!


	@staticmethod
	def compatible_candidates(source, bounds, comparator=iou, agreement_threshold=0.9):
		for item in source:
			assert 'implication' in item, 'Because of the implication'
			agreement = comparator(*bounds, *item['implication'])
			if agreement >= agreement_threshold:
				yield item

	@staticmethod
	def balanced_choose(gen, options: Sequence[str], counts: Dict[str, int] = None, n=None):
		if counts is None:
			counts = {}

		if isinstance(options, dict):
			keys = list(options.keys())
		else:
			keys = options

		wts = {x: max(1, counts.get(x, 1)) for x in options}
		wts = {x: 1 / w for x, w in wts.items()}
		wts = np.asarray([wts[x] for x in options])
		wts /= wts.sum()

		picks = gen.choice(keys, size=(n or 1), p=wts)
		if isinstance(options, dict):
			return options[picks[0]] if n is None else [options[x] for x in picks]
		return picks[0] if n is None else picks

	def generate_ambiguous_premise(self, story, term, prior, *, strict=False):
		if strict:
			yield from self.compatible_candidates(self.generate_ambiguous_premise(story, term, prior, strict=False),
			                                      bounds=prior, agreement_threshold=self.agreement_threshold,
			                                      comparator=self.agreement_score)
		else:
			yield from generate_ambiguous_evidence(story, term, prior, gen=self.gen,
			                                       agreement_threshold=self.agreement_threshold)
			yield from generate_ambiguous_evidence_from_value(story, term, prior, gen=self.gen,
			                                                  agreement_threshold=self.agreement_threshold)


	def generate_interval_premise(self, story, term, prior, *, strict=False):
		if strict:
			yield from self.compatible_candidates(generate_interval_evidence(story, term, prior, gen=self.gen,
			                                                                 agreement_threshold=self.agreement_threshold),
			                                      bounds=prior, agreement_threshold=self.agreement_threshold,
			                                      comparator=self.agreement_score)
		else:
			yield from generate_interval_evidence(story, term, prior, gen=self.gen)


	def generate_precise_premise(self, story, term, prior):
		yield from generate_precise_evidence(story, term, prior, gen=self.gen,
		                                     agreement_threshold=self.agreement_threshold)


	def generate_premises(self, story, terms: Dict[str, Tuple[float, float]], *, strict=False,
	                      spec_key='specificity', hist_key='premises'):
		premises = {}

		for term, prior in terms.items():
			population = list(self.spawn_premises(story, term, prior, strict=strict))

			assert len(population) > 0, "No premises available for term %s in story %s" % (term, story['ID'])

			spec = self.balanced_choose(self.gen, [premise['type'] for premise in population],
			                            self.history.setdefault(spec_key, {}))
			pick = self.balanced_choose(self.gen, {premise['ID']: premise for premise in population if premise['type'] == spec},
			                            self.history.setdefault(hist_key, {}))
			pick['term'] = term
			# pick['prior'] = prior
			premises[term] = pick

		return premises


	def generate_alternate_premises(self, story, terms: Dict[str, Tuple[float, float]], *, hist_key='alternates'):
		groups = {}

		for term, prior in terms.items():
			ambiguous = list(self.generate_ambiguous_premise(story, term, prior, strict=True))

			picks = ambiguous if len(ambiguous) < self.num_alternates \
				else self.balanced_choose(self.gen, {p['ID']: p for p in ambiguous}, self.history.setdefault(hist_key, {}),
			                             n=self.num_alternates)
			groups.setdefault('ambiguous', {}).setdefault(term, []).extend(picks)

			interval = list(self.generate_interval_premise(story, term, prior, strict=True))
			picks = interval if len(interval) < self.num_alternates \
				else self.balanced_choose(self.gen, {p['ID']: p for p in interval}, self.history.setdefault(hist_key, {}),
			                             n=self.num_alternates)
			groups.setdefault('interval', {}).setdefault(term, []).extend(picks)

			precise = list(self.generate_precise_premise(story, term, prior))
			picks = precise if len(precise) < self.num_alternates \
				else self.balanced_choose(self.gen, {p['ID']: p for p in precise}, self.history.setdefault(hist_key, {}),
			                             n=self.num_alternates)
			groups.setdefault('precise', {}).setdefault(term, []).extend(picks)

		for spec, group in groups.items():
			for term, premises in group.items():
				for premise in premises:
					premise['term'] = term
					# premise['prior'] = terms[term]
					premise['agreement'] = self.agreement_score(*terms[term], *premise['implication'])

		return groups


	def compute_commonsense(self, premises, commonsense):
		for premise in premises:
			lc, uc = commonsense[premise['term']]
			premise['commonsense'] = self.commonsense_score(lc, uc, *premise['implication'])
			premise['prior'] = [lc, uc]


	@staticmethod
	def commonsense_score(lc, uc, start, end):
		return beta_agreement_score(lc, uc, start, end)


	@staticmethod
	def agreement_score(lc, uc, start, end):
		return iou(lc, uc, start, end)


	def optimize_params(self, system, winner_init, loser_init, fixed=None):
		optim = ATEGapOptimization(system, bounds1=winner_init, bounds2=loser_init, fixed1=fixed, fixed2=fixed,)
		                           # gen=self.gen)
		winner_param, loser_param = optim.solve()

		winner = {'params': winner_param}
		winner['ate'] = [v.item() for v in optim.estimand_bounds1(winner_param)]

		loser = {'params': loser_param}
		loser['ate'] = [v.item() for v in optim.estimand_bounds2(loser_param)]

		return winner, loser


	def generate_questions(self, winner, loser):
		yield from self.operator.verbalize_questions(winner, loser)


	def generate(self, N=100, pbar=False):
		itr = tqdm(total=N) if pbar else None

		completed = 0
		while completed < N:
			q = {}

			# select story
			winner, loser = self.select_story()
			q['ID'], q['loser-ID'] = winner['ID'], loser['ID']
			itr.set_description(f'Generating {q["ID"]}')

			# select fixed
			fixed = self.select_fixed(winner, loser)
			remaining = [term for term in winner['system'].ate_dofs if term not in fixed]
			q['fixed'] = fixed

			# generate fixed
			winner_premises = self.generate_premises(winner, {term: winner['commonsense'].get(term, [0, 1]) for term in fixed})
			loser_premises = self.generate_premises(loser, {term: loser['commonsense'].get(term, [0, 1]) for term in fixed})

			# optimize remaining
			winner_commonsense, loser_commonsense = winner['commonsense'], loser['commonsense']
			winner_init, loser_init = winner_commonsense.copy(), loser_commonsense.copy()
			winner_init.update({term: premise['implication'] for term, premise in winner_premises.items()})
			loser_init.update({term: premise['implication'] for term, premise in loser_premises.items()})

			winner_info, loser_info = self.optimize_params(winner['system'], winner_init, loser_init, fixed=fixed)
			q['winner'], q['loser'] = winner_info, loser_info

			# verbalize remaining
			winner_premises.update(self.generate_premises(winner, {term: winner_info['params'][term] for term in remaining}, strict=True))
			loser_premises.update(self.generate_premises(loser, {term: loser_info['params'][term] for term in remaining}, strict=True))

			# generate alternates
			winner_alternates = self.generate_alternate_premises(winner, winner_info['params'])
			winner_info['alternates'] = winner_alternates
			loser_alternates = self.generate_alternate_premises(loser, loser_info['params'])
			loser_info['alternates'] = loser_alternates

			# compute commonsense # skipped -> too slow (do as needed)
			if not self.skip_commonsense_scores:
				self.compute_commonsense(winner_premises.values(), winner_commonsense)
				self.compute_commonsense(loser_premises.values(), loser_commonsense)
				for group in winner_alternates.values():
					for premises in group.values():
						self.compute_commonsense(premises, winner_commonsense)
				for group in loser_alternates.values():
					for premises in group.values():
						self.compute_commonsense(premises, loser_commonsense)

			winner_info['premises'] = list(winner_premises.values())
			loser_info['premises'] = list(loser_premises.values())

			# verbalize graph info
			q['graph'] = list(winner['system'].verbalize_graph(winner))

			if 'intro' in winner:
				q['intro'] = winner['intro']

			# verbalize question + answer
			q['questions'] = list(self.generate_questions(winner, loser))

			if itr is not None:
				itr.update(1)
			completed += 1
			self.update_history(q)
			yield q



def test_generate():

	G = Generator(seed=0).prepare()


	qs = list(G.generate(1, pbar=False))

	# save_json(qs, util.data_root() / 'arg-ate-questions.json')

	print(qs)



@fig.script('gen')
def generate_questions(config):

	outpath = config.pull('out', str(util.data_root() / 'questions.json'))
	if outpath is not None:
		outpath = Path(outpath)
		print(f'Writing prompts to {outpath}')

	seed = config.pull('seed', 11)
	skip_commonsense_scores = config.pull('skip-scores', False)

	G = Generator(seed=seed, skip_commonsense_scores=skip_commonsense_scores).prepare()

	N = config.pull('n', 10)
	safe = config.pull('safe', False)
	pbar = config.pull('pbar', True)

	q_gen = G.generate(N, pbar=pbar and not safe)

	if safe:
		qs = []
		itr = tqdm(total=N) if pbar else None
		try:
			for _ in range(N):
				q = next(q_gen)
				qs.append(q)
				if itr is not None:
					itr.update(1)
		except KeyboardInterrupt:
			raise
		except:
			print(f'Error encountered, saving {len(qs)} questions...')
			save_json(qs, outpath)
			raise
	else:
		qs = list(q_gen)

	save_json(qs, outpath)

	print(f'Saving {len(qs)} questions')




