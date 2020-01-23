import concurrent.futures
import gym
import errno
import itertools
import os
import os.path as osp
import pickle
import time

import numpy as np
from scipy.spatial import Delaunay
import scipy.stats as stats
from sklearn.neighbors import NearestNeighbors as knn
import tensorflow as tf

from .controller import Controller
from .safe_set import SafeSet, create_ss_new_goal
from .value import ValueFunc
import copy


def get_preds(acs, env_name, state):
	local_env = gym.make(env_name, cem_env=True)
	local_env.reset()
	local_env.set_state(state)

	for i, a in enumerate(acs):
		local_env.step(a)

	pred_states = np.array(local_env.get_hist())
	pred_costs = np.array(local_env.get_costs())

	return (pred_states, pred_costs)

def get_preds_parallel(all_acs, env_name, state):
	with concurrent.futures.ProcessPoolExecutor() as executor:
		f_list = [executor.submit(get_preds, all_acs[p1], env_name, state) for p1 in range(all_acs.shape[0])]
		return [f.result() for f in f_list]


class LMPC(Controller):

	def __init__(self, cfg):
		self.env_name = cfg.env_name
		self.SS = []
		self.value_funcs = []
		self.all_safe_states = []
		self.value_ss_approx_models = []
		self.soln_mode = cfg.soln_mode
		self.ss_approx_mode = cfg.ss_approx_mode
		self.variable_start_state = cfg.variable_start_state
		self.variable_start_state_cost = cfg.variable_start_state_cost
		self.n_samples_start_state_opt = cfg.n_samples_start_state_opt
		self.start_state_opt_success_thresh = cfg.start_state_opt_success_thresh
		self.ss_value_train_success_thresh = cfg.ss_value_train_success_thresh
		self.update_SS_and_value_func_CEM = cfg.update_SS_and_value_func_CEM
		self.max_update_SS_value = cfg.max_update_SS_value

		self.model_logdir = osp.join(cfg.save_dir, cfg.model_logdir)

		if not os.path.exists(self.model_logdir):
			os.makedirs(self.model_logdir)

		if self.ss_approx_mode == "knn":
			self.ss_approx_model = knn(n_neighbors=1)
		elif self.ss_approx_mode == "convex_hull":
			pass
		else:
			raise Exception("Unsupported SS Approx Mode")

		self.value_approx_mode = cfg.value_approx_mode

		if self.soln_mode == "cem":
			self.optimizer_params = cfg.optimizer_params
			self.alpha_thresh = cfg.alpha_thresh
			self.ac_lb = cfg.ac_lb
			self.ac_ub = cfg.ac_ub
			self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.optimizer_params["plan_hor"]])
			self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.optimizer_params["plan_hor"]])
			self.dU = self.ac_lb.size
			self.cem_env = gym.make(self.env_name, cem_env=True)
			self.ac_buf = np.array([]).reshape(0, self.dU)
			self.parallelize_cem = cfg.parallelize_cem

	def act(self, state):
		if self.soln_mode == "cem":
			return self.cem_act(state)
		elif self.soln_mode == "exact":
			raise Exception("Not Implemented")
		else:
			raise Exception("Unsupported solution method")

	def cem_act(self, state):
		if self.ac_buf.shape[0] > 0:
			action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
			return action

		soln, rollout = self.run_cem(state, mean=self.prev_sol, var=self.init_var)
		self.prev_sol = np.concatenate([np.copy(soln)[self.optimizer_params["per"]*self.dU:], np.zeros(self.optimizer_params["per"]*self.dU)])
		self.ac_buf = soln[:self.optimizer_params["per"]*self.dU].reshape(-1, self.dU)
		return self.cem_act(state)

	def reset(self):
		self.SS = []
		self.value_funcs = []
		self.all_safe_states = []
		self.value_ss_approx_models = []
		if self.soln_mode == "cem":
			self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.optimizer_params["plan_hor"]])

	# Returns start state for next iteration
	def train(self, samples):
		self.SS.append(SafeSet())
		self.value_funcs.append(ValueFunc(self.value_approx_mode))

		for s in samples:
			self.SS[-1].add_sample(s)
			self.value_funcs[-1].add_sample(s)

		# First fit safety density models (TODO: fix this)
		if self.ss_approx_mode == "convex_hull":
			value_ss_approx_model = Delaunay(list(itertools.chain.from_iterable(self.SS[-1].state_data)))
			self.value_ss_approx_models.append(value_ss_approx_model)
		elif self.ss_approx_mode == "knn":
			value_ss_approx_model = knn(n_neighbors=1)
			value_ss_approx_model.fit(list(itertools.chain.from_iterable(self.SS[-1].state_data)))
			self.value_ss_approx_models.append(value_ss_approx_model)	
		else:
			raise Exception("Unsupported SS Approx Mode")

		# Update safety model
		all_safe_states = list(itertools.chain.from_iterable([s.state_data for s in self.SS]))
		self.all_safe_states= list(itertools.chain.from_iterable(all_safe_states))

		if self.ss_approx_mode == "knn":
			self.ss_approx_model.fit(np.array(self.all_safe_states))
		elif self.ss_approx_mode == "convex_hull":
			self.ss_approx_model = Delaunay(self.all_safe_states)
		else:
			raise Exception("Unsupported SS Approx Mode")

		# Fit value function
		self.value_funcs[-1].fit()

	def compute_valid_start_state(self, desired_start=None):
		# Sample some start state in safe set, compute self.n_samples_start_state_opt. samples for it
		# and resample if less than self.start_state_opt_success_thresh % are successful
		valid_start = None

		if self.variable_start_state and self.variable_start_state_cost == "towards":
			print("DESIRED START", desired_start)
			# TODO: make this less hacky by just taking normal euclidean distance, including velocities...
			sorted_all_safe_states = sorted( self.all_safe_states, key=lambda x: np.linalg.norm( np.array([x[0], x[2]]) - np.array([desired_start[0], desired_start[2]]) ) )
			# sorted_all_safe_states = sorted( self.all_safe_states, key=lambda x: np.linalg.norm(x - desired_start) )

		# Find a valid start state
		if self.variable_start_state:
			i = 0
			valid = False
			while not valid:
				if self.variable_start_state_cost != "towards":
					sampled_start = self.all_safe_states[np.random.randint(len(self.all_safe_states))]
				else:
					sampled_start = sorted_all_safe_states[i]

				print("SAMPLED START", sampled_start)
				 # TODO: parallelize later if needed
				valid_starts = []
				valid_trajs = []
				for _ in range(self.n_samples_start_state_opt):
					traj, traj_valid = self.traj_opt(sampled_start, desired_start)
					if traj_valid:
						valid_starts.append(traj[-self.optimizer_params["plan_hor"]])
						valid_trajs.append(traj)

				print("NUM VALIDS", len(valid_starts))
				if len(valid_starts) >= int(self.start_state_opt_success_thresh * self.n_samples_start_state_opt):
					valid = True 
					valid_start = valid_starts[np.random.randint(len(valid_starts))]

				i += 1

			print("VALID START", valid_start)

		return valid_start

	def traj_opt(self, obs, desired_start=None):
		if self.soln_mode == "cem":
			mean = np.tile((self.ac_lb + self.ac_ub) / 2, [self.optimizer_params["plan_hor"]+self.optimizer_params["extra_hor"]])
			var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.optimizer_params["plan_hor"]+self.optimizer_params["extra_hor"]])
			soln, pred_traj = self.run_cem(obs, mean, var, traj_opt_mode=True, desired_start=desired_start)
			invalid = int(self.unsafe(pred_traj[-1:], self.ss_approx_model)[0, 0]) # Check whether terminal state is actually safe
		elif self.soln_mode == "exact":
			raise Exception("Not Implemented")
		else:
			raise Exception("Unsupported solution method")

		return pred_traj, 1 - invalid

	def set_goal(self, goal_state):
		if self.cem_env.goal_state == goal_state:
			return
		assert 0
		self.cem_env.set_goal(goal_state)
		goal_fn = self.cem_env.goal_fn
		new_values = []
		for value in self.value_funcs:
			new_values.append(create_value_function_new_goal(value, goal_fn))
		self.value_funcs = new_values
		for value in self.value_funcs:
			value.fit()
		new_ss = []
		for ss in self.SS:
			new_ss.append(create_ss_new_goal(ss, goal_fn))
		self.SS = new_ss
		self.value_ss_approx_models =[]
		for ss in self.SS:
			if self.ss_approx_mode == "convex_hull":
				value_ss_approx_model = Delaunay(list(itertools.chain.from_iterable(ss.state_data)))
				self.value_ss_approx_models.append(value_ss_approx_model)
			elif self.ss_approx_mode == "knn":
				value_ss_approx_model = knn(n_neighbors=1)
				value_ss_approx_model.fit(list(itertools.chain.from_iterable(ss.state_data)))
				self.value_ss_approx_models.append(value_ss_approx_model)	
			else:
				raise("Unsupported SS Approx Mode")

		all_safe_states = list(itertools.chain.from_iterable([s.state_data for s in self.SS]))
		self.all_safe_states= list(itertools.chain.from_iterable(all_safe_states))

		if self.ss_approx_mode == "knn":
			self.ss_approx_model.fit(np.array(self.all_safe_states))
		elif self.ss_approx_mode == "convex_hull":
			self.ss_approx_model = Delaunay(self.all_safe_states)
		else:
			raise("Unsupported SS Approx Mode")


	def run_cem(self, obs, mean, var, traj_opt_mode=False, desired_start=None):
		if traj_opt_mode:
			plan_hor = self.optimizer_params["plan_hor"] + self.optimizer_params["extra_hor"]
		else:
			plan_hor = self.optimizer_params["plan_hor"]

		lb = np.tile(self.ac_lb, [plan_hor])
		ub = np.tile(self.ac_ub, [plan_hor])
		X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))
		for i in range(self.optimizer_params["num_iters"]):
			lb_dist, ub_dist = mean - lb, ub - mean
			constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)
			samples = X.rvs(size=[self.optimizer_params["popsize"], plan_hor*self.dU]) * np.sqrt(constrained_var) + mean
			if traj_opt_mode:
				costs, rollouts = self._predict_and_eval_expansion(obs, samples, desired_start=desired_start)
			else:
				costs, rollouts = self._predict_and_eval(obs, samples)
			costs = costs.reshape(self.optimizer_params["npart"], self.optimizer_params["popsize"]).T.mean(1)
			# print(" CEM Iteration ", i, "Cost: ", np.mean(costs))
			elites = samples[np.argsort(costs)][:self.optimizer_params["num_elites"]]
			min_costs = np.sort(costs)[:self.optimizer_params["num_elites"]]
			# print("MAX MIN COST: ", np.max(min_costs))

			new_mean = np.mean(elites, axis=0)
			new_var = np.var(elites, axis=0)
			mean, var = self.optimizer_params["alpha"] * mean + (1 - self.optimizer_params["alpha"]) * new_mean, self.optimizer_params["alpha"] * var + (1 - self.optimizer_params["alpha"]) * new_var # refit mean/var
		
		# Return best action and corresponding full rollout
		return elites[0], rollouts[np.argmin(costs)]

	def unsafe(self, states, approx_model):
		if self.ss_approx_mode == "convex_hull":
			return (approx_model.find_simplex(states)<0).astype(int)

		elif self.ss_approx_mode == "knn":
			dists = approx_model.kneighbors(states)[0]
			unsafe_idxs = np.where(dists > self.alpha_thresh)
			safe_idxs = np.where(dists <= self.alpha_thresh)
			dists[unsafe_idxs] = 1
			dists[safe_idxs] = 0
			return dists
		else:
			raise Exception("Unsupported SS Approx Mode")

	def compute_nn(self, states, approx_model):
		dists = approx_model.kneighbors(states)
		nearest_neighbors = [self.all_safe_states[i[0]] for i in dists[1]]
		return np.array(nearest_neighbors)

	def compute_value(self, states):
		# Now evaluate queried states on each value_ss_approx_model and blow
		# up values accordingly if needed
		value_mat = []
		for value_func, value_ss_approx_model in zip(self.value_funcs, self.value_ss_approx_models):
			res = self.unsafe(states, value_ss_approx_model)
			values = value_func.value(states) + 1e6 * res # blow up unsafe values
			value_mat.append(value_func.value(states))
		value_mat = np.vstack(value_mat)
		return np.min(value_mat, axis=0)


	def _predict_and_eval(self, obs, ac_seqs, get_pred_trajs=True):
		"""
		Takes in Numpy arrays obs (current image) and action sequences to evaluate. Returns the predicted
		frames and the costs associated with each action sequence.
		"""
		ac_seqs = np.reshape(ac_seqs, [-1, self.optimizer_params["plan_hor"], self.dU])

		# Keep setting state to obs and simulating actions: # TODO: parallelize
		if not self.parallelize_cem:
			start = time.perf_counter()	
			pred_trajs, costs = self.cem_env.vectorized_step(obs, ac_seqs)
			finish = time.perf_counter()
			# print("Time Taken Serially: " + str(round(finish-start, 2)))
		else:
			start = time.perf_counter()
			results = get_preds_parallel(ac_seqs, self.env_name, obs)
			pred_trajs = np.array([r[0] for r in results])
			costs = np.array([r[1] for r in results])
			finish = time.perf_counter()
			# print("Time Taken Parallelized: " + str(round(finish-start, 2)))

		# Compute safe trajectories, if enough are safe in this CEM Iter, then
		# save states and VALUE adjusted costs for THOSE trajectories so that
		# they can be added to safe set and value buffer

		safety_check = self.unsafe(pred_trajs[:, -1], self.ss_approx_model)
		traj_value = self.compute_value(pred_trajs[:, -1])

		costs_cumsum = np.apply_along_axis(lambda x: np.cumsum(np.array(x)[::-1])[::-1], 1, costs)
		reshaped_value = traj_value.reshape((-1, 1))
		value_targets = costs_cumsum + reshaped_value
		orig_costs = copy.deepcopy(costs) 

		costs = np.sum(costs, axis=1)
		safety_check = safety_check.flatten()
		costs += traj_value
		costs += safety_check * 1e6

		if self.update_SS_and_value_func_CEM:
			# Update data for safe set and value func
			success_percentage =  (len(safety_check) - np.sum(safety_check))/len(safety_check) 
			if success_percentage > self.ss_value_train_success_thresh:
				# Get successful idxs
				success_idxs = (1 - np.array(safety_check)).astype(bool)
				value_targets_filtered = value_targets[success_idxs]
				costs_filtered = orig_costs[success_idxs]
				pred_trajs_filtered = pred_trajs[success_idxs]
				pred_trajs_filtered = pred_trajs_filtered[:,:-1,:]

				state_train_data = pred_trajs_filtered.reshape((-1, pred_trajs_filtered.shape[-1]))
				value_train_data = value_targets_filtered.flatten()
				cost_train_data = costs_filtered.flatten()
				s = {"states": state_train_data[:self.max_update_SS_value], "values": value_train_data[:self.max_update_SS_value], "costs": cost_train_data[:self.max_update_SS_value]}
				# Add samples to most recent safe set and value_func train set
				self.SS[-1].add_sample(s)
				self.value_funcs[-1].add_sample(s)

		return costs, pred_trajs

	def _predict_and_eval_expansion(self, obs, ac_seqs, get_pred_trajs=True, desired_start=None):
		"""
		Takes in Numpy arrays obs (current image) and action sequences to evaluate. Returns the predicted
		frames and the costs associated with each action sequence.
		"""
		ac_seqs = np.reshape(ac_seqs, [-1, self.optimizer_params["plan_hor"]+self.optimizer_params["extra_hor"], self.dU])

		# Keep setting state to obs and simulating actions: # TODO: parallelize
		if not self.parallelize_cem:
			start = time.perf_counter()	
			pred_trajs, costs = self.cem_env.vectorized_step(obs, ac_seqs)
			finish = time.perf_counter()
			# print("Time Taken Serially: " + str(round(finish-start, 2)))
		else:
			start = time.perf_counter()
			results = get_preds_parallel(ac_seqs, self.env_name, obs)
			pred_trajs = np.array([r[0] for r in results])
			costs = np.array([r[1] for r in results])
			finish = time.perf_counter()
			# print("Time Taken Parallelized: " + str(round(finish-start, 2)))

		if self.variable_start_state_cost == "indicator":
			start_state_opt_costs = 1 - self.unsafe(pred_trajs[:, 0], self.ss_approx_model) # 0 cost if unsafe
			for i in range(1, pred_trajs.shape[1]-1):
				start_state_opt_costs += (1 - self.unsafe(pred_trajs[:, i], self.ss_approx_model) )
		elif self.variable_start_state_cost == "nearest_neighbor":
			# Expand away from nearest neighbor
			# Find nearest neighbors of all points in safe set and encourage being far away
			# Don't include first point since it has to be in the safeset
			start_state_opt_costs = np.zeros(len(pred_trajs[:, 0]))
			for i in range(1, pred_trajs.shape[1]-1):
				nn = self.compute_nn(pred_trajs[:, i], self.ss_approx_model)
				start_state_opt_costs += -np.sum((pred_trajs[:, i] - self.compute_nn(pred_trajs[:, i], self.ss_approx_model))**2, axis=1)
		elif self.variable_start_state_cost == "towards":
			# Expand toward desired_start
			start_state_opt_costs = np.zeros(len(pred_trajs[:, 0]))
			for i in range(1, pred_trajs.shape[1]-1):
				start_state_opt_costs += np.sum((pred_trajs[:, i] - desired_start)**2, axis=1)
		else:
			raise Exception("Unsupported Start State Selection Cost Function")

		traj_value = self.compute_value(pred_trajs[:, -1])
		costs_cumsum = np.apply_along_axis(lambda x: np.cumsum(np.array(x)[::-1])[::-1], 1, costs)
		reshaped_value = traj_value.reshape((-1, 1))
		value_targets = costs_cumsum + reshaped_value 

		start_state_opt_costs = start_state_opt_costs.flatten()
		safety_check = self.unsafe(pred_trajs[:, -1], self.ss_approx_model)
		safety_check = safety_check.flatten()
		start_state_opt_costs += safety_check * 1e6

		# Update data for safe set and value func
		if self.update_SS_and_value_func_CEM:
			success_percentage =  (len(safety_check) - np.sum(safety_check))/len(safety_check) 
			if success_percentage > self.ss_value_train_success_thresh:
				# Get successful idxs
				success_idxs = (1 - np.array(safety_check)).astype(bool)
				value_targets_filtered = value_targets[success_idxs]
				costs_filtered = costs[success_idxs]
				pred_trajs_filtered = pred_trajs[success_idxs]
				pred_trajs_filtered = pred_trajs_filtered[:,:-1,:]

				state_train_data = pred_trajs_filtered.reshape((-1, pred_trajs_filtered.shape[-1]))
				value_train_data = value_targets_filtered.flatten()
				cost_train_data = costs_filtered.flatten()

				s = {"states": state_train_data[:self.max_udpate_SS_value], "values": value_train_data[:self.max_udpate_SS_value], "costs": cost_train_data[:self.max_udpate_SS_value]}
				# Add samples to most recent safe set and value_func train set
				self.SS[-1].add_sample(s)
				self.value_funcs[-1].add_sample(s)

		return start_state_opt_costs, pred_trajs

	def save_controller_state(self):
		pickle.dump(self.ss_approx_model, open(os.path.join(self.model_logdir, "ss_approx_model.pkl"), "wb"))
		# self.all_safe_states
		pickle.dump(self.all_safe_states, open(os.path.join(self.model_logdir, "all_safe_states.pkl"), "wb"))
		# Save data for self.SS
		pickle.dump([s.get_data() for s in self.SS], open(os.path.join(self.model_logdir, "ss_data.pkl"), "wb"))

		# Save self.value_funcs
		for i, v in enumerate(self.value_funcs):
			value_model = v.model
			desired_path = os.path.join(self.model_logdir, "value", "value_model_" + str(i))
			if not os.path.exists(desired_path):
				os.makedirs(desired_path)
			if v.approx_mode == 'pe':
				value_model.save(desired_path)
			else:
				pickle.dump(value_model, open(osp.join(desired_path, "model.pkl"), "wb"))

		# Save data for self.value_funcs
		pickle.dump([v.get_data() for v in self.value_funcs], open(os.path.join(self.model_logdir, "value_data.pkl"), "wb"))

		# Save self.value_ss_approx_models
		desired_path = osp.join(self.model_logdir, "value_ss")
		if not os.path.exists(desired_path):
			os.makedirs(desired_path)
		pickle.dump(self.value_ss_approx_models, open(osp.join(desired_path, "value_ss_approx_models.pkl"), "wb"))

	
	def restore_controller_state(self, model_logdir=None):
		if model_logdir is not None:
			self.model_logdir = model_logdir
		# Reload safe set model and data
		self.ss_approx_model = pickle.load( open(os.path.join(self.model_logdir, "ss_approx_model.pkl"), "rb") )
		self.all_safe_states = pickle.load( open(os.path.join(self.model_logdir, "all_safe_states.pkl"), "rb") )

		safe_set_data = pickle.load(open(os.path.join(self.model_logdir, "ss_data.pkl"), "rb"))
		self.SS = [SafeSet(state_data=state_data) for state_data in safe_set_data]

		value_models_base_dir = os.path.join(self.model_logdir, "value")
		self.value_funcs = []
		value_func_data = pickle.load(open(os.path.join(self.model_logdir, "value_data.pkl"), "rb"))
		value_folder_list = [folder for folder in os.listdir(value_models_base_dir) if folder.startswith("value")]
		value_folder_list.sort(key=lambda value_folder: int(value_folder.split('_')[-1]))

		g = tf.Graph()
		with g.as_default():
			for i, value_folder in enumerate(value_folder_list):
				state_data = value_func_data[i][0]
				value_data = value_func_data[i][1]
				cost_data = value_func_data[i][2]
				self.value_funcs.append(ValueFunc(self.value_approx_mode, load_model=True, model_dir=os.path.join(value_models_base_dir, value_folder), state_data=state_data, value_data=value_data, cost_data=cost_data))

		self.value_ss_approx_models = pickle.load( open(os.path.join(self.model_logdir, "value_ss", "value_ss_approx_models.pkl"), "rb") )

