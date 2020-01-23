# TODO: add goal conditioned filter or something
from .utils import process_sample_for_goal

def create_ss_new_goal(ss_old, goal_fn):
	data = ss_old.get_data()
	state_data, cost_data = [], []
	for i in range(len(data[0])):
		sample = {
			'states': data[0][i]
		}
		new_sample = process_sample_for_goal(sample, goal_fn)
		state_data.append(new_sample['states'])
	return SafeSet(state_data=state_data)


class SafeSet:
	def __init__(self, state_data=()):
		self.state_data = list(state_data) # nsamples X horizon X state_dim

	def get_samples(self, t=None):
		if idx is None:
			return np.array(self.state_data)
		else:
			return np.array(self.state_data)[:,t,:]

	def add_sample(self, sample):
		if sample['successful']:
			self.state_data.append(sample['states'])

	def load_data(self, state_data):
		self.state_data = state_data

	def get_data(self):
		return self.state_data




