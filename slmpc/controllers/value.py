import itertools

from dotmap import DotMap
import numpy as np
from sklearn.neighbors import NearestNeighbors as knn
from sklearn.linear_model import Ridge
import tensorflow as tf

from slmpc.modeling.layers import FC
from slmpc.modeling.models import BNN

def value_pe_constructor(sess):
    model = BNN(DotMap(
        name="value", num_networks=5,
        sess=sess, load_model=False,
        model_dir="models"
    ))
    model.add(FC(500, input_dim=4, activation='swish', weight_decay=0.0001))
    model.add(FC(500, activation='swish', weight_decay=0.00025))
    model.add(FC(500, activation='swish', weight_decay=0.00025))
    model.add(FC(1, weight_decay=0.0005))
    model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001}, suffix = "val")
    return model

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
		elif approx_mode == "pe":
			self.sess = tf.Session()
			# TODO: store value func params in a dotmap config
			self.model = value_pe_constructor(self.sess)
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
		elif self.approx_mode == "pe":
			self.model.train(self.state_fit_data, self.value_fit_data[...,np.newaxis], epochs=20)
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
		elif self.approx_mode == "pe":
			return self.model.predict(states, factored=False)[0].squeeze() # Ignore variance for now
		else:
			raise("Unsupported value approximation mode")






