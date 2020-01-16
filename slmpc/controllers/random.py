import numpy as np

from .controller import Controller
from .safe_set import SafeSet

class RandomController(Controller):

	def __init__(self, env):
		self.env = env

	def act(self, state):
		# TODO: set up and solve problem in cvxpy
		return self.env.sample()

	def reset(self):
		self.env.reset()

	def train(self, samples):
		return
