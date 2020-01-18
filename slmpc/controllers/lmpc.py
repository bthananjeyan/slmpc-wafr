import numpy as np

from .controller import Controller
from .safe_set import SafeSet
import scipy.stats as stats
import itertools
from sklearn.neighbors import NearestNeighbors as knn
from scipy.spatial import ConvexHull

import concurrent.futures
import gym
import time

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
		self.env = cfg.env
		self.env_name = self.env.env_name
		self.SS = []
		self.act_fn = ACT_FNS[self.env.name]
		self.soln_mode = cfg.soln_mode
		self.ss_approx_mode = cfg.ss_approx_mode

		if self.ss_approx_mode == "knn":
			self.ss_approx_model = knn(n_neighbors=1)
		elif self.ss_approx_mode == "convex_hull":
			pass
		else:
			raise("Unsupported SS Approx Mode")

		if self.soln_mode == "cem":
			self.optimizer_params = {"num_iters": 5, "popsize": 200, "npart": 1, "num_elites": 40, "plan_hor": 20, "per": 1, "alpha": 0.1}
			self.alpha_thresh = cfg.alpha_thresh
			self.ac_lb = self.env.action_space.low
			self.ac_ub = self.env.action_space.high
			self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.optimizer_params["plan_hor"]])
			self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.optimizer_params["plan_hor"]])
			self.dU = self.ac_lb.size
			self.cem_env = cfg.cem_env
			self.ac_buf = np.array([]).reshape(0, self.dU)
			self.parallelize_cem = cfg.parallelize_cem

	def act(self, state):
		if self.soln_mode == "cem":
			return self.cem_act(state)
		elif self.soln_mode == "exact":
			# TODO: set up and solve problem in cvxpy
			return self.act_fn(state, self.env)
		else:
			raise Exception("Unsupported solution method")

	def cem_act(self, state):
		if self.ac_buf.shape[0] > 0:
			action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
			return action

		soln = self.run_cem(state, mean=self.prev_sol, var=self.init_var)
		self.prev_sol = np.concatenate([np.copy(soln)[self.optimizer_params["per"]*self.dU:], np.zeros(self.optimizer_params["per"]*self.dU)])
		self.ac_buf = soln[:self.optimizer_params["per"]*self.dU].reshape(-1, self.dU)
		return self.cem_act(state)

	def reset(self):
		self.SS = []
		self.env.reset()
		if self.soln_mode == "cem":
			self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.optimizer_params["plan_hor"]])

	def train(self, samples):
		self.SS.append(SafeSet(self.env.horizon))
		for s in samples:
			self.SS[-1].add_sample(s)

		# Take union of all safesets 
		all_states = list(itertools.chain.from_iterable([s.state_data for s in self.SS]))
		all_states= list(itertools.chain.from_iterable(all_states))

		if self.ss_approx_mode == "knn":
			self.ss_approx_model.fit(np.array(all_states))
		elif self.ss_approx_mode == "convex_hull":
			self.ss_approx_model = ConvexHull(all_states)
		else:
			raise("Unsupported SS Approx Mode")

	def run_cem(self, obs, mean, var):
		lb = np.tile(self.ac_lb, [self.optimizer_params["plan_hor"]])
		ub = np.tile(self.ac_ub, [self.optimizer_params["plan_hor"]])
		X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))
		for i in range(self.optimizer_params["num_iters"]):
			lb_dist, ub_dist = mean - lb, ub - mean
			constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)
			samples = X.rvs(size=[self.optimizer_params["popsize"], self.optimizer_params["plan_hor"]*self.dU]) * np.sqrt(constrained_var) + mean
			costs, rollouts = self._predict_and_eval(obs, samples)
			costs = costs.reshape(self.optimizer_params["npart"], self.optimizer_params["popsize"]).T.mean(1)
			print(" CEM Iteration ", i, "Cost: ", np.mean(costs))
			elites = samples[np.argsort(costs)][:self.optimizer_params["num_elites"]]
			min_costs = np.sort(costs)[:self.optimizer_params["num_elites"]]
			print("MAX MIN COST: ", np.max(min_costs))

			new_mean = np.mean(elites, axis=0)
			new_var = np.var(elites, axis=0)
			mean, var = self.optimizer_params["alpha"] * mean + (1 - self.optimizer_params["alpha"]) * new_mean, self.optimizer_params["alpha"] * var + (1 - self.optimizer_params["alpha"]) * new_var # refit mean/var
		return mean

	def unsafe(self, states):
		if self.ss_approx_mode == "convex_hull":
			raise("Not Implemented Error")
		elif self.ss_approx_mode == "knn":
			dists = self.ss_approx_model.kneighbors(states)[0]
			unsafe_idxs = np.where(dists > self.alpha_thresh)
			safe_idxs = np.where(dists <= self.alpha_thresh)
			dists[unsafe_idxs] = 1
			dists[safe_idxs] = 0
			return dists
		else:
			raise("Unsupported SS Approx Mode")


	def _predict_and_eval(self, obs, ac_seqs, get_pred_trajs=True):
		"""
		Takes in Numpy arrays obs (current image) and action sequences to evaluate. Returns the predicted
		frames and the costs associated with each action sequence.
		"""
		ac_seqs = np.reshape(ac_seqs, [-1, self.optimizer_params["plan_hor"], self.dU])

		# Keep setting state to obs and simulating actions: # TODO: parallelize
		if not self.parallelize_cem:
			start = time.perf_counter()	
			costs = np.zeros((self.optimizer_params["popsize"], self.optimizer_params["plan_hor"]))
			pred_trajs = np.zeros((self.optimizer_params["popsize"], self.optimizer_params["plan_hor"]+1, len(obs)))
			for i in range(self.optimizer_params["popsize"]):
				self.cem_env.reset()
				self.cem_env.set_state(obs)
				for j in range(self.optimizer_params["plan_hor"]):
					self.cem_env.step(ac_seqs[i, j])

				pred_trajs[i] = np.array(self.cem_env.get_hist())
				costs[i] = np.array(self.cem_env.get_costs())
			finish = time.perf_counter()
			# print("Time Taken Serially: " + str(round(finish-start, 2)))
		else:
			start = time.perf_counter()
			results = get_preds_parallel(ac_seqs, self.env_name, obs)
			pred_trajs = np.array([r[0] for r in results])
			costs = np.array([r[1] for r in results])
			finish = time.perf_counter()
			# print("Time Taken Parallelized: " + str(round(finish-start, 2)))

		safety_check = self.unsafe(pred_trajs[:, -1])
		costs = np.sum(costs, axis=1)
		safety_check = safety_check.flatten()
		costs += safety_check * 1e6

		return costs, pred_trajs

def pointbot_act(state, env):
	return env.sample()

def cartpole_act(state, env):
	return env.sample()

ACT_FNS = {
	'pointbot': pointbot_act,
	'cartpole': cartpole_act
}
