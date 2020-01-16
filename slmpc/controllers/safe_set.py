
class SafeSet:

	def __init__(self, horizon):
		self.state_data = [] # nsamples X horizon X state_dim
		self.cost_data = [] # nsamples X horizon

	def get_samples(self, t=None):
		if idx is None:
			return np.array(self.state_data)
		else:
			return np.array(self.state_data)[:,t,:]

	def add_sample(self, sample):
		self.state_data.append(sample['states'])
		self.cost_data.append(sample['costs'])

