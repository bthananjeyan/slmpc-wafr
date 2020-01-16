import numpy as np

from .controller import Controller
from .safe_set import SafeSet

class LMPC(Controller):

	def __init__(self, env):
		self.env = env
		self.SS = []

	def act(self, state):
		# TODO: set up and solve problem in cvxpy
		raise NotImplementedError

	def reset(self):
		self.SS = []
		self.env.reset()

	def train(self, samples):
		self.SS.append(SafeSet(self.env.horizon))
		for s in samples:
			self.SS.add_sample(s)

