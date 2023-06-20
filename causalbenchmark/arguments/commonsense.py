from .imports import *
from .stories import iterate_scenarios, get_available_stories, get_story_system
from .queries import create_query


def compute_beta_from_confidence_interval(conf_l, conf_u, conf_l_level=0.05, conf_u_level=0.95, *,
                                          ensure_mode=True, threshold=1e-4):
	x0 = np.ones((2,)) * 10.
	
	def step(x):
		alpha, beta = x
		d = stats.beta(alpha, beta)
		error = (d.ppf(conf_l_level) - conf_l) ** 2 + (d.ppf(conf_u_level) - conf_u) ** 2
		if error < threshold:
			return 0.
		return error

	res = opt.minimize(step, x0, constraints=opt.LinearConstraint(np.eye(2), 1.1, np.inf))
	
	alpha, beta = res.x.tolist()
	return alpha, beta


def iou(lb1, ub1, lb2, ub2):
	'''intersection over union'''
	
	li = max(lb1, lb2)
	ui = min(ub1, ub2)
	
	if li > ui:
		return 0.
	
	lu = min(lb1, lb2)
	uu = max(ub1, ub2)
	
	return (ui - li) / (uu - lu)


def beta_agreement_score(lc, uc, start, end, *, eps=1e-5, threshold=1e-4):
	if (uc - lc) < eps:
		lc, uc = lc - eps / 2, uc + eps / 2

	alpha, beta = compute_beta_from_confidence_interval(lc, uc, threshold=threshold)
	prior = stats.beta(alpha, beta)
	mode = (alpha - 1) / (alpha + beta - 2)
	mx_val = prior.pdf(mode)

	start = max(start, 0.)
	end = min(end, 1.)

	if (end - start) < eps:
		return prior.pdf(start) / mx_val
	return ((1 if np.isclose(end, 1.) else prior.cdf(end)) - (0. if np.isclose(start, 0.) else prior.cdf(start))) \
		/ (end - start) / mx_val



def simple_containment(lc, uc, start, end):
	return max(0., min((min(uc, end) - max(lc, start)) / (end - start), 1.))



def commonsense_score(commonsense, params, *, eps=1e-5):
	if commonsense is None:
		raise NotImplementedError
	
	scores = {}
	
	for key, lims in params.items():
		if key in commonsense:
			scores[key] = beta_agreement_score(*commonsense[key], *lims)
	
	return scores


def test_ate_commonsense_score():
	# random.seed(0)
	
	stories = list(get_available_stories())
	full = [list(iterate_scenarios(story)) for story in stories]
	full = [scens for scens in full if len(scens) > 0 and scens[0]['query'] == 'ate']
	
	scenarios = random.choice(full)
	
	assert len(scenarios) > 1
	
	picks = random.choice(scenarios, size=2, replace=False)
	
	init1 = picks[0]['commonsense']
	init2 = picks[1]['commonsense']
	
	system = get_story_system(scenarios[0])
	
	optim = create_query('ate', system=system, bounds1=init1, bounds2=init2)
	
	bounds1, bounds2 = optim.solve()
	
	scores1 = commonsense_score(scenarios[0]['commonsense'], bounds1)
	scores2 = commonsense_score(scenarios[1]['commonsense'], bounds2)
	
	assert len(scores1) == len(scores2)


def test_med_commonsense_score():
	# random.seed(0)
	
	stories = list(get_available_stories())
	full = [list(iterate_scenarios(story)) for story in stories]
	full = [scens for scens in full if len(scens) > 0 and scens[0]['query'] == 'med']
	
	scenarios = random.choice(full)
	
	assert len(scenarios) > 1
	
	pick = random.choice(scenarios)
	
	system = get_story_system(pick)
	
	init = pick['commonsense']
	
	optim = create_query('med', system=system, bounds=init, direct_is_higher=True)
	
	bounds1, bounds2 = optim.solve()
	
	scores1 = commonsense_score(pick['commonsense'], bounds1)
	scores2 = commonsense_score(pick['commonsense'], bounds2)
	
	assert len(scores1) == len(scores2)





