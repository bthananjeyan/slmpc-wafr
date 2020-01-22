# TODO: add goal conditioned filter or something
from .utils import process_sample_for_goal

def create_ss_new_goal(ss_old, goal_fn):
	data = ss_old.get_data()
	state_data, cost_data = [], []
	for i in range(len(data[0])):
		sample = {
			'states': data[0][i],
			'costs': data[1][i]
		}
		new_sample = process_sample_for_goal(sample, goal_fn)
		state_data.append(new_sample['states'])
		cost_data.append(new_sample['costs'])
	return SafeSet(state_data=state_data, cost_data=cost_data)


class SafeSet:
	def __init__(self, state_data=[], cost_data=[]):
		self.state_data = state_data # nsamples X horizon X state_dim
		self.cost_data = cost_data # nsamples X horizon

	def get_samples(self, t=None):
		if idx is None:
			return np.array(self.state_data)
		else:
			return np.array(self.state_data)[:,t,:]

	def add_sample(self, sample):
		self.state_data.append(sample['states'])
		self.cost_data.append(sample['costs'])

	def load_data(self, state_data, cost_data):
		self.state_data = state_data
		self.cost_data = cost_data

	def get_data(self):
		return (self.state_data, self.cost_data)




