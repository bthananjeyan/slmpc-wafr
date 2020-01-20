import numpy as np

from .controller import Controller
from .safe_set import SafeSet

class RandomController(Controller):

	def __init__(self, env):
		self.env = cfg.env

	def act(self, state):
		return self.env.sample()

	def reset(self):
		self.env.reset()

	def train(self, samples):
		return
