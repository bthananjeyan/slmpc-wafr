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

def get_sample(start_state, exp_cfg, goal_state=None):
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
	if goal_state is not None:
		local_env.set_goal(goal_state)
		local_controller.cem_env.set_goal(goal_state)
	obs = local_env.reset()

	if start_state is not None: # multi-start case
		local_env.set_state(start_state)

	data['states'].append(obs)
	done = False
	while not done:
		action = local_controller.act(obs)
		obs, cost, done, _ = local_env.step(action)
		data['states'].append(obs)
		data['actions'].append(action)
		data['costs'].append(cost)
	data['total_cost'] = np.sum(data['costs'])
	data['values'] = np.cumsum(data['costs'][::-1])[::-1]
	data['successful'] = data['costs'][-1] == 0
	return data


def get_samples_parallel(valid_starts, exp_cfg, goal_state):
	with concurrent.futures.ProcessPoolExecutor() as executor:
		f_list = [executor.submit(get_sample, valid_starts[p1], exp_cfg, goal_state) for p1 in range(len(valid_starts))]
		return [f.result() for f in f_list]

class Experiment:

	def __init__(self, env, exp_cfg):
		self.env = env
		self.exp_cfg = exp_cfg
		self.samples_per_iteration = self.exp_cfg.samples_per_iteration
		self.num_iterations = self.exp_cfg.num_iterations
		self.parallelize_rollouts = exp_cfg.parallelize_rollouts

		self.goal_schedule = exp_cfg.goal_schedule
		self.desired_starts = exp_cfg.desired_starts
		self.variable_start_state_cost = exp_cfg.variable_start_state_cost

		self.log_all_data = exp_cfg.log_all_data
		self.save_dir = self.exp_cfg.save_dir
		self.demo_path = self.exp_cfg.demo_path
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
		self.exp_cfg.save_dir = self.save_dir = osp.join(self.save_dir, datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))
		os.makedirs(self.save_dir)

		if exp_cfg.controller_type == "random":
			self.controller = RandomController(exp_cfg)
		elif exp_cfg.controller_type == "lmpc_expect":
			self.controller = LMPC(exp_cfg)
		else:
			raise Exception("Unsupported controller.")

		with open(os.path.join(self.save_dir, "config.txt"), "wb") as f:
			pickle.dump(self.exp_cfg, f)

	def reset(self):
		self.all_samples = []
		self.all_valid_starts = []
		self.mean_costs = []
		self.cost_stds = []

	def dump_logs(self):
		np.save(osp.join(self.save_dir, "mean_costs.npy"), self.mean_costs)
		if self.log_all_data:
			with open(osp.join(self.save_dir, "samples.pkl"), "wb") as f:
				pickle.dump({"samples": self.all_samples, "valid_starts": self.all_valid_starts}, f)

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
			print(len(data['states']), obs, action, cost)
			data['states'].append(obs)
			data['actions'].append(action)
			data['costs'].append(cost)
		data['total_cost'] = np.sum(data['costs'])
		data['values'] = np.cumsum(data['costs'][::-1])[::-1]
		data['successful'] = data['costs'][-1] == 0
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
				'values' : demo_full_data[i]["values"],
				'successful': True}
			demo_samples.append(demo_data)

		self.all_samples.append(demo_samples)
		self.controller.train(demo_samples)
		for i in range(self.num_iterations):
			print("##### Iteration %d #####"%i)
			self.controller.set_goal(self.goal_schedule(i))
			self.controller.save_controller_state()
			self.env.set_goal(self.goal_schedule(i))

			if not self.parallelize_rollouts:
				samples = []
				valid_starts = []

				# Same start state for all samples per iteration
				if self.variable_start_state_cost == "towards":
					valid_start = self.controller.compute_valid_start_state(self.desired_starts[i])
				else:
					valid_start = self.controller.compute_valid_start_state()

				for _ in range(self.samples_per_iteration):
					samples.append(self.sample(valid_start))
					valid_starts.append(valid_start)
			else:
				# Same start state for all samples per iteration
				if self.variable_start_state_cost == "towards":
					valid_start = self.controller.compute_valid_start_state(self.desired_starts[i])
				else:
					valid_start = self.controller.compute_valid_start_state()

				valid_starts = [valid_start for _ in range(self.samples_per_iteration)]
				samples = get_samples_parallel(valid_starts, self.exp_cfg, self.goal_schedule(i))

			self.all_samples.append(samples)
			self.all_valid_starts.append(valid_starts)

			mean_cost = np.mean([s['total_cost'] for s in samples])
			self.mean_costs.append(mean_cost)
			self.cost_stds.append(np.std([s['total_cost'] for s in samples]))
			print("Average Cost: %f"%mean_cost)
			print("Individual Costs:")
			print([s['total_cost'] for s in samples])
			print([s['states'][-1] for s in samples])

			self.controller.train(samples)
			self.controller.save_controller_state()
			self.dump_logs()
		self.plot_results(save_file=osp.join(self.save_dir, "costs.png"), show=False)

	@property
	def stats(self):
		return np.array(self.mean_costs), np.add(self.mean_costs, self.cost_stds), np.subtract(self.mean_costs, self.cost_stds)

	def plot_results(self, save_file=None, show=True):
		def plot_mean_and_CI(mean, ub, lb, color_mean=None, color_shading=None):
			# plot the shaded range of the confidence intervals
			plt.fill_between(range(mean.shape[0]), ub, lb,
							 color=color_shading, alpha=.5)
			# plot the mean on top
			plt.plot(mean, color_mean)
		import matplotlib.pyplot as plt
		mean, ub, lb = self.stats
		plot_mean_and_CI(mean, ub, lb, 'b', 'b')
		plt.title("Mean Trajectory Cost vs. Iteration")
		plt.xlabel("Iteration")
		plt.ylabel("Trajectory Cost")
		if save_file is not None:
			plt.savefig(save_file)
		if show:
			plt.show()
		plt.clf()
