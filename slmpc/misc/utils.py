class Schedule:
	"""
	Schedule wrapper.
	"""

	def __init__(self, switchpoints, outputs):
		raise NotImplementedError

	def __call__(self, t):
		raise NotImplementedError

class SingleSwitchSchedule(Schedule):
	"""
	Schedule with a single switch point
	"""

	def __init__(self, switchpoint, outputs):
		assert len(outputs) == 2
		self.switchpoint = switchpoint
		self.outputs = outputs

	def __call__(self, t):
		return self.outputs[int(t >= self.switchpoint)]

class NoSwitchSchedule(Schedule):
	"""
	Schedule with a single switch point
	"""

	def __init__(self, switchpoint, output):
		self.output = output

	def __call__(self, t):
		return self.output
