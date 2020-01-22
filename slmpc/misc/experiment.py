from datetime import datetime
import itertools
import os
import os.path as osp
import pickle
import concurrent.futures
import gym
import time
import numpy as np

from slmpc.controllers import LMPC, RandomController

def get_sample(start_state, exp_cfg):
	data = {
		'states': [],
		'actions': [],
		'costs': []
	}

	local_env = gym.make(exp_cfg.env_name)
	if exp_cfg.controller_type == "random":
		local_controller = RandomController(exp_cfg)
	elif exp_cfg.controller_type == "lmpc_expect":
		local_controller = LMPC(exp_cfg)
	else:
		raise Exception("Unsupported controller.")

	local_controller.restore_controller_state()
	obs = local_env.reset()

	if start_state is not None: # multi-start case
		local_env.set_state(start_state)

	data['states'].append(obs)
	done = False
	while not done:
		print("ACTED")
		action = local_controller.act(obs)
		obs, cost, done, _ = local_env.step(action)
		data['states'].append(obs)
		data['actions'].append(action)
		data['costs'].append(cost)
	data['total_cost'] = np.sum(data['costs'])
	data['values'] = np.cumsum(data['costs'][::-1])[::-1]
	return data


def get_samples_parallel(valid_starts, exp_cfg):
	with concurrent.futures.ProcessPoolExecutor() as executor:
		f_list = [executor.submit(get_sample, valid_starts[p1], exp_cfg) for p1 in range(len(valid_starts))]
		return [f.result() for f in f_list]

class Experiment:

	def __init__(self, env, exp_cfg):
		if exp_cfg.controller_type == "random":
			self.controller = RandomController(exp_cfg)
		elif exp_cfg.controller_type == "lmpc_expect":
			self.controller = LMPC(exp_cfg)
		else:
			raise Exception("Unsupported controller.")

		self.env = env
		self.exp_cfg = exp_cfg
		self.samples_per_iteration = self.exp_cfg.samples_per_iteration
		self.num_iterations = self.exp_cfg.num_iterations
		self.parallelize_rollouts = exp_cfg.parallelize_rollouts
		self.desired_starts = exp_cfg.desired_starts
		self.variable_start_state_cost = exp_cfg.variable_start_state_cost

		self.log_all_data = exp_cfg.log_all_data
		self.save_dir = self.exp_cfg.save_dir
		self.demo_path = self.exp_cfg.demo_path
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
		self.save_dir = osp.join(self.save_dir, datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))
		os.makedirs(self.save_dir)

	def reset(self):
		self.all_samples = []
		self.mean_costs = []

	def dump_logs(self):
		np.save(osp.join(self.save_dir, "mean_costs.npy"), self.mean_costs)
		if self.log_all_data:
			with open(osp.join(self.save_dir, "samples.pkl"), "wb") as f:
				pickle.dump(self.all_samples, f)

	def sample(self, start_state):
		data = {
			'states': [],
			'actions': [],
			'costs': []
		}
		obs = self.env.reset()

		if start_state is not None: # multi-start case
			self.env.set_state(start_state)

		data['states'].append(obs)
		done = False
		while not done:
			action = self.controller.act(obs)
			obs, cost, done, _ = self.env.step(action)
			data['states'].append(obs)
			data['actions'].append(action)
			data['costs'].append(cost)
		data['total_cost'] = np.sum(data['costs'])
		data['values'] = np.cumsum(data['costs'][::-1])[::-1]
		return data

	# TODO: build something to create visualizations of safe set
	# I would put this in the safe_set file.
	def run(self):
		self.reset()
		# First train on demos
		demo_full_data = pickle.load(open(self.demo_path, "rb"))

		demo_samples = []
		for i in range(len(demo_full_data)):
			demo_data = {
				'states': demo_full_data[i]["obs"],
				'actions': demo_full_data[i]["ac"],
				'costs': demo_full_data[i]["costs"],
				'total_cost': demo_full_data[i]["cost_sum"],
				'values' : demo_full_data[i]["values"]
			}
			demo_samples.append(demo_data)

		self.all_samples.append(demo_samples)
		self.controller.train(demo_samples)
		self.controller.save_controller_state()

		for i in range(self.num_iterations):
			print("##### Iteration %d #####"%i)

			if not self.parallelize_rollouts:
				samples = []
				for _ in range(self.samples_per_iteration):
					if self.variable_start_state_cost == "towards":
						valid_start = self.controller.compute_valid_start_state(self.desired_starts[i])
					else:
						valid_start = self.controller.compute_valid_start_state()
					samples.append(self.sample(valid_start))
			else:
				if self.variable_start_state_cost == "towards":
					valid_starts = [self.controller.compute_valid_start_state(self.desired_starts[i]) for _ in range(self.samples_per_iteration)]
				else:
					valid_starts = [self.controller.compute_valid_start_state() for _ in range(self.samples_per_iteration)]
				samples = get_samples_parallel(valid_starts, self.exp_cfg)

			self.all_samples.append(samples)
			self.controller.train(samples)
			self.controller.save_controller_state()

			mean_cost = np.mean([s['total_cost'] for s in samples])
			self.mean_costs.append(mean_cost)
			print("Average Cost: %f"%mean_cost)

			self.dump_logs()
		self.plot_results(save_file=osp.join(self.save_dir, "costs.png"), show=False)

	@property
	def stats(self):
		return self.mean_costs

	def plot_results(self, save_file=None, show=True):
		import matplotlib.pyplot as plt
		plt.plot(self.stats)
		plt.title("Mean Trajectory Cost vs. Iteration")
		plt.xlabel("Iteration")
		plt.ylabel("Trajectory Cost")
		if save_file is not None:
			plt.savefig(save_file)
		if show:
			plt.show()
		plt.clf()
