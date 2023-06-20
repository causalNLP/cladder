from .. import util


def create_system(key, *args, **kwargs):
	return _known_systems[key](*args, **kwargs)



_known_systems = {}
class CausalSystem:
	def __init_subclass__(cls, **kwargs):
		super().__init_subclass__(**kwargs)
		name = getattr(cls, 'name', None)
		if name is not None:
			_known_systems[name] = cls


	def graph_edges(self):
		raise NotImplementedError


	_graph_templates = None
	def verbalize_graph(self, story, *, gen=None):
		if self._graph_templates is None:
			raise NotImplementedError

		template_order = list(self._graph_templates.keys())
		if gen is not None:
			gen.shuffle(template_order)

		for key in template_order:
			template = self._graph_templates[key]
			line = util.pformat(template, **story)
			line = line[0].upper() + line[1:]
			yield {'template': key, 'verb': line, 'ID': key}



class ConfoundingSystem(CausalSystem):
	name = 'confounding'
	ate_dofs = r'\sum_{U=v} P(U=v)*[P(Y=1|U=v,X=1) - P(Y=1|U=v, X=0)]'
	ate_dofs = ['U', 'Y|X=0,U=0', 'Y|X=1,U=0', 'Y|X=0,U=1', 'Y|X=1,U=1']

	@staticmethod
	def ate_fast(x):
		c = x[0]
		y00, y10, y01, y11 = x[1:]
		return c * (y11 - y01) + (1 - c) * (y10 - y00)
	
	
	_graph_templates = {
		'casual': '{Xnode} has an effect on {Ynode}.',
		'natural': '{Xnode} leads to {Ynode}, but there are also other factors.',
		'technical': '{Xnode} leads to {Ynode} with a confounder {Unode}.',
		'formal': '{Xnode} is a direct cause of {Ynode} with a confounder of {Unode}. '
		          'Treat all variables as binary and assume no other variables or causal relationships.',
	}
	def graph_edges(self):
		return [
			['U', 'X'],
			['U', 'Y'],
			['X', 'Y'],
		]
	


class IVSystem(CausalSystem):
	name = 'IV'
	ate_str = '[P(Y=1|V2=1)-P(Y=1|V2=0)]/[P(X=1|V2=1)-P(X=1|V2=0)]'
	ate_dofs = ['Y|Z=1', 'Y|Z=0', 'X|Z=1', 'X|Z=0']

	@staticmethod
	def ate_fast(x):
		y1, y0, x1, x0 = x
		return (y1 - y0) / (x1 - x0 + 1e-8)


	_graph_templates = {
		'casual': '{Znode} can be used to better understand the effect of {Xnode} on {Ynode}.',
		'natural': '{Znode} leads to {Xnode} and {Xnode} leads to {Ynode}.',
		'technical': '{Xnode} leads to {Ynode} with an instrumental variable {Znode} to deal with confounding.',
		'formal': '{Xnode} is a direct cause of {Ynode} with a confounder {Unode} and an instrumental variable {Znode}. '
		          'Treat all variables as binary and assume no other variables or causal relationships.',
	}
	def graph_edges(self):
		return [
			['U', 'X'],
			['U', 'Y'],
			['Z', 'X'],
			['X', 'Y'],
		]



class FrontdoorSystem(CausalSystem):
	name = 'frontdoor'
	ate_str = r'\sum_{V3 = v} [P(V3 = v|X = 1) - P(V3 = v|X = 0)] * [\sum_{X = h} P(Y = 1|X = h,V3 = v)*P(X = h)]'
	ate_dofs = ['X', 'M|X=1', 'M|X=0', 'Y|X=0,M=0', 'Y|X=1,M=0', 'Y|X=0,M=1', 'Y|X=1,M=1']

	@staticmethod
	def ate_fast(x):
		x, v31, v30, y00, y10, y01, y11 = x
		return (v31 - v30) * (x * (y11 - y10) + (1 - x) * (y01 - y00))


	_graph_templates = {
		'casual': '{Xnode} has an effect on {Ynode} by {Mnode}.',
		'natural': '{Xnode} leads to {Mnode} and {Mnode} leads to {Ynode}, but there may be other factors as well.',
		'technical': '{Xnode} leads to {Ynode} with a mediator {Mnode}, but {Xnode} and {Ynode} are also confounded.',
		'formal': '{Xnode} is a cause of {Ynode} through the mediator {Mnode} and {Xnode} and {Ynode} are confounded by {Unode}. '
		          'Treat all variables as binary and assume no other variables or causal relationships.',
	}
	def graph_edges(self):
		return [
			['U', 'X'],
			['U', 'Y'],
			['X', 'M'],
			['M', 'Y'],
		]



class MediationSystem(CausalSystem):
	name = 'mediation'
	ate_str = 'P(Y=1|X=1) - P(Y=1|X=0)'
	ate_dofs = ['Y|X=0', 'Y|X=1']

	@staticmethod
	def ate_fast(x):
		y0, y1 = x
		return y1 - y0


	nde_str = '\sum_{V2=v} P(V2=v|X=0)*[P(Y=1|X=1,V2=v) - P(Y=1|X=0, V2=v)]'
	nde_dofs = ['M|X=0', 'Y|X=0,M=0', 'Y|X=1,M=0', 'Y|X=0,M=1', 'Y|X=1,M=1']

	@staticmethod
	def nde_fast(x):
		v20, y00, y10, y01, y11 = x
		return v20 * (y11 - y01) + (1 - v20) * (y10 - y00)


	nie_str = '\sum_{V2 = v} P(Y=1|X =0,V2 = v)*[P(V2 = v | X = 1) − P(V2 = v | X = 0)]'
	nie_dofs = ['Y|X=0,M=0', 'Y|X=0,M=1', 'M|X=1', 'M|X=0']

	@staticmethod
	def nie_fast(x):
		y00, y01, v21, v20 = x
		return (y01 - y00) * (v21 - v20)


	# _graph_template = {
	# 	'dry': '{X} directly causes {Y} and indirectly also through by {M}.',
	# }
	def graph_edges(self):
		return [
			['X', 'M'],
			['M', 'Y'],
			['X', 'Y'],
		]








