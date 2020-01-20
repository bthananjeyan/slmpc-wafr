import itertools
from sklearn.neighbors import NearestNeighbors as knn
from sklearn.linear_model import Ridge
import numpy as np

# TODO: add goal conditioned filter or something
class ValueFunc:
	def __init__(self, approx_mode):
		self.state_data = [] # nsamples X horizon X state_dim
		self.cost_data = [] # nsamples X horizon
		self.value_data = [] # nsamples X horizon
		self.approx_mode = approx_mode
		self.model_fit = False
		if approx_mode == "linear":
			self.model = Ridge(alpha=0) 
		elif approx_mode == "knn":
			self.model = knn(n_neighbors=5) # TODO: think about n_neighbors
		else:
			raise("Unsupported value approximation mode")

	def get_samples(self, t=None):
		if idx is None:
			return np.array(self.state_data)
		else:
			return np.array(self.state_data)[:,t,:]

	def add_sample(self, sample):
		self.state_data.append(sample['states'])
		self.cost_data.append(sample['costs'])
		self.value_data.append(sample['values'])

	def fit(self):
		self.value_fit_data = np.concatenate(self.value_data, axis=0)
		state_fit_data = [s[:-1] for s in self.state_data]
		self.state_fit_data = np.concatenate(state_fit_data, axis=0)

		if self.approx_mode == "linear":
			self.model.fit(self.state_fit_data, self.value_fit_data)
		elif self.approx_mode == "knn":
			self.model.fit(self.state_fit_data)
		else:
			raise("Unsupported value approximation mode")

		self.model_fit = True

	def value(self, states):
		assert(self.model_fit)

		if self.approx_mode == "linear":
			return self.model.predict(states)
		elif self.approx_mode == "knn":
			neighbors = self.model.kneighbors(states, return_distance=False)
			neighbor_values = self.value_fit_data[neighbors]	
			return np.mean(neighbor_values, axis=1)
		else:
			raise("Unsupported value approximation mode")






