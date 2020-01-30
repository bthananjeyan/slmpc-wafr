from datetime import datetime
import itertools
import os
import os.path as osp
import pickle
import concurrent.futures
import gym
import time
import numpy as np
import scipy.io as sio
from slmpc.controllers import LMPC, RandomController

def load_reacher_samples(demo_load_path, env, lower=0, upper=100, max_num_samples=100):
    mat = sio.loadmat(demo_load_path)
    print(mat.keys())
    demo_samples = []
    for i, acs in enumerate(mat['actions']):
        demo_data = {}
        demo_data['states'] = mat['observations'][i] # TODO: maybe exclude goal from obs?
        demo_data['actions'] = acs
        demo_data['costs'] = env.post_process(mat['observations'][i], acs, mat['rewards'][i])
        demo_data['total_cost'] = np.sum(demo_data['costs'])
        demo_data['values'] = np.cumsum(demo_data['costs'][::-1])[::-1]
     
        # Check if total cost is in reasonable range and that the goal was achieved at the end
        if demo_data['values'][0] < upper and demo_data['values'][0] > lower and demo_data['costs'][-1] == 0:
            demo_data['successful'] = True
            if len(demo_samples) < max_num_samples:
                demo_samples.append(demo_data)

    return demo_samples


def get_sample(start_state, exp_cfg, goal_state=None):
	data = {
		'states': [],
		'actions': [],
		'costs': []
	}

	print("GOAL STATE", goal_state)
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
		if exp_cfg.name == "pendulum":
			local_env.set_state(local_env.state_from_obs(start_state))
		else:
			local_env.set_state(start_state)

	data['states'].append(obs)
	done = False
	data['collision'] = False
	while not done:
		action = local_controller.act(obs)
		obs, cost, done, _ = local_env.step(action)
		# if exp_cfg.has_obstacles:
		# 	print(len(data['states']), obs, local_env.collision_check(obs), action, cost)
		# else:
		# print(len(data['states']), obs, action, cost)
		data['states'].append(obs)
		data['actions'].append(action)
		data['costs'].append(cost)
		if exp_cfg.has_obstacles and local_env.collision_check(obs):
			print("COLLIDED!")
			data['collision'] = True
	data['total_cost'] = np.sum(data['costs'])
	data['values'] = np.cumsum(data['costs'][::-1])[::-1]
	data['successful'] = int(data['costs'][-1]) == 0
	return data


def get_samples_parallel(valid_starts, exp_cfg, goal_state):
	with concurrent.futures.ProcessPoolExecutor() as executor:
		f_list = [executor.submit(get_sample, valid_starts[p1], exp_cfg, goal_state) for p1 in range(len(valid_starts))]
		return [f.result() for f in f_list]

class Experiment:

	def __init__(self, env, exp_cfg):
		self.env = env
		self.name = exp_cfg.name 
		self.exp_cfg = exp_cfg
		self.samples_per_iteration = self.exp_cfg.samples_per_iteration
		self.num_iterations = self.exp_cfg.num_iterations
		self.parallelize_rollouts = exp_cfg.parallelize_rollouts
		self.has_obstacles = exp_cfg.has_obstacles
		self.env.has_obstacles = self.has_obstacles
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

		with open(os.path.join(self.save_dir, "config.pkl"), "wb") as f:
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
			if self.name == "pendulum":
				self.env.set_state(self.env.state_from_obs(start_state))
			else:
				self.env.set_state(start_state)

		data['states'].append(obs)
		done = False
		data['collision'] = False 
		while not done:
			action = self.controller.act(obs)
			obs, cost, done, _ = self.env.step(action)
			# if self.has_obstacles:
			# 	print(len(data['states']), obs, self.env.collision_check(obs), action, cost)
			# else:
			# print(len(data['states']), obs, action, cost)
			data['states'].append(obs)
			data['actions'].append(action)
			data['costs'].append(cost)
			if self.has_obstacles and self.env.collision_check(obs):
				print("COLLIDED")
				data['collision'] = True
		data['total_cost'] = np.sum(data['costs'])
		data['values'] = np.cumsum(data['costs'][::-1])[::-1]
		data['successful'] = int(data['costs'][-1]) == 0
		return data

	def run(self):
		self.reset()
		# First train on demos
		if self.name == "reacher":
			demo_samples = load_reacher_samples(self.demo_path, self.env)
		else:
			demo_full_data = pickle.load(open(self.demo_path, "rb"))

			demo_samples = []
			for i in range(len(demo_full_data)):
				demo_data = {
					'states': demo_full_data[i]["obs"],
					'actions': demo_full_data[i]["ac"],
					'costs': demo_full_data[i]["costs"],
					'total_cost': demo_full_data[i]["cost_sum"],
					'values' : demo_full_data[i]["values"],
					'successful': True,
					'collision': False}
				demo_samples.append(demo_data)

		self.all_samples.append(demo_samples)
		self.controller.train(demo_samples)
		for i in range(self.num_iterations):
			print("##### Iteration %d #####"%i)
			print("GOAL SCHEDULE", self.goal_schedule(i))
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
