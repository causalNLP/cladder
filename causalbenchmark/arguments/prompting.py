from .imports import *

class Prompter:
	@staticmethod
	def prompt_keys(system, story):
		raise NotImplementedError
	
	@classmethod
	def prompt_params(cls, system, story):
		keys = cls.prompt_keys(system, story)
		
		lines = []
		for key in keys:
			lines.append(f'p({key}) = ?')
		
		return '\n'.join(lines)



class ATE_Prompter(Prompter):
	@staticmethod
	def prompt_keys(system, story):
		dofs = system.ate_dofs
		
		keys = []
		for dof in dofs:
			v, *other = dof.split('|')
			# v = v + '=1'
			if len(other):
				v = v + '|' + '|'.join(other)
			keys.append(v)
		
		return keys



class Mediation_Prompter(Prompter):
	@staticmethod
	def prompt_keys(system, story):
		return sorted(list(set(system.nde_dofs) | set(system.nie_dofs)))





