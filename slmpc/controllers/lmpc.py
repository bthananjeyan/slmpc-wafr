import numpy as np

from .controller import Controller
from .safe_set import SafeSet

class LMPC(Controller):

	def __init__(self, env):
		self.env = env
		self.SS = []
		self.act_fn = ACT_FNS[self.env.name]

	def act(self, state):
		# TODO: set up and solve problem in cvxpy
		return self.act_fn(state, self.env)

	def reset(self):
		self.SS = []
		self.env.reset()

	def train(self, samples):
		self.SS.append(SafeSet(self.env.horizon))
		for s in samples:
			self.SS[-1].add_sample(s)


def pointbot_act(state, env):
	return env.sample()

ACT_FNS = {
	'pointbot': pointbot_act
}
