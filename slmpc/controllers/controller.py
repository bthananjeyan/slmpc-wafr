
class Controller:

	def act(self, state):
		raise NotImplementedError

	def reset(self):
		raise NotImplementedError

	def train(self, samples):
		raise NotImplementedError
