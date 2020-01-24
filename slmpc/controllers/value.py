import itertools
import os.path as osp

from dotmap import DotMap
import numpy as np
from sklearn.neighbors import NearestNeighbors as knn
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
import tensorflow as tf

from slmpc.controllers.utils import process_sample_for_goal
from slmpc.modeling.layers import FC
from slmpc.modeling.models import BNN

def value_pe_constructor(sess, dO, load_model, model_dir):
	model = BNN(DotMap(
		name="value", num_networks=5,
		sess=sess, load_model=load_model,
		model_dir=model_dir
	))
	if not load_model:
		model.add(FC(500, input_dim=dO, activation='swish', weight_decay=0.0001))
		model.add(FC(500, activation='swish', weight_decay=0.00025))
		model.add(FC(500, activation='swish', weight_decay=0.00025))
		model.add(FC(1, weight_decay=0.0005, activation='ReLU'))

	model.finalize(tf.compat.v1.train.AdamOptimizer, {"learning_rate": 0.001}, suffix = "val")
	return model

def create_value_function_new_goal(v_old, goal_fn, dO):
	assert 0
	data = v_old.get_data()
	state_data, value_data, cost_data = [], [], []
	for i in range(len(data[0])):
		sample = {
			'states': data[0][i],
			'costs': data[2][i]
		}
		new_sample = process_sample_for_goal(sample, goal_fn)
		state_data.append(new_sample['states'])
		value_data.append(new_sample['values'])
		cost_data.append(new_sample['costs'])
	return ValueFunc(v_old.approx_mode, dO, load_model=False, model_dir=None, state_data=state_data, value_data=value_data, cost_data=cost_data)

# TODO: add goal conditioned filter or something
class ValueFunc:
	def __init__(self, approx_mode, dO, load_model=False, model_dir=None, state_data=(), value_data=(), cost_data=()):
		self.state_data = list(state_data) # nsamples X horizon X state_dim
		self.value_data = list(value_data) # nsamples X horizon
		self.cost_data = list(cost_data) # nsamples X horizon
		self.approx_mode = approx_mode
		self.dO = dO
		self.model_fit = load_model
		if approx_mode == "linear":
			self.model = Ridge(alpha=0) 
			self.model = KernelRidge(alpha=1, kernel='polynomial', coef0=4)
			# self.model = MLPRegressor(hidden_layer_sizes = (100, 100), activation = 'relu', solver = 'sgd', learning_rate = 'adaptive')
			if load_model:
				self.model = pickle.load(open(osp.join(model_dir, "model.pkl"), "rb"))
		elif approx_mode == "knn":
			self.model = knn(n_neighbors=5)
		elif approx_mode == "pe":
			self.graph = tf.Graph()
			with self.graph.as_default():
				self.sess = tf.compat.v1.Session()
				# TODO: store value func params in a dotmap config
				self.model = value_pe_constructor(self.sess, self.dO, load_model, model_dir)
		else:
			raise("Unsupported value approximation mode")


	def get_samples(self, t=None):
		if idx is None:
			return np.array(self.state_data)
		else:
			return np.array(self.state_data)[:,t,:]

	def add_sample(self, sample):
		self.state_data.append(sample['states'])
		self.value_data.append(sample['values'])
		self.cost_data.append(sample['costs'])

	def load_data(self, state_data, value_data, cost_data):
		self.state_data = state_data
		self.value_data = value_data
		self.cost_data = cost_data

	def get_data(self):
		return (self.state_data, self.value_data, self.cost_data)

	def fit(self):
		self.model_fit = True
		self.value_fit_data = np.concatenate(self.value_data, axis=0)
		state_fit_data = [s[:-1] for s in self.state_data]
		self.state_fit_data = np.concatenate(state_fit_data, axis=0)

		if self.approx_mode == "linear":
			# import IPython; IPython.embed()
			self.model.fit(self.state_fit_data, self.value_fit_data)
		elif self.approx_mode == "knn":
			self.model.fit(self.state_fit_data)
		elif self.approx_mode == "pe":
			TD = False
			self.model.train(self.state_fit_data, self.value_fit_data[...,np.newaxis], epochs=10)
			if TD:
				def td_iteration():
					self.next_state_fit_data = np.concatenate([s[1:] for s in self.state_data], axis=0)
					self.cost_fit_data = np.concatenate(self.cost_data, axis=0)
					targets = self.value(self.next_state_fit_data) + self.cost_fit_data
					self.model.train(self.state_fit_data, targets[...,np.newaxis], epochs=10)
				for i in range(5):
					td_iteration()
		else:
			raise("Unsupported value approximation mode")


	def value(self, states):
		assert(self.model_fit)

		if self.approx_mode == "linear":
			return 0.001 * self.model.predict(states)
		elif self.approx_mode == "knn":
			neighbors = self.model.kneighbors(states, return_distance=False)
			neighbor_values = self.value_fit_data[neighbors]	
			return np.mean(neighbor_values, axis=1)
		elif self.approx_mode == "pe":
			return self.model.predict(states, factored=False)[0].squeeze() # Ignore variance for now
		else:
			raise("Unsupported value approximation mode")

	def visualize(self):
		assert(self.model_fit)
		states = []
		for i in range(-50, 50):
			for j in range(-50, 50):
				states.append([i, 0, j, 0])
		states = np.array(states)
		values = self.value(states)
		import matplotlib.pyplot as plt
		plt.imshow(values.reshape((100, 100)))
		plt.show()





