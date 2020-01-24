import multiprocessing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from dotmap import DotMap

from slmpc.misc.experiment import Experiment
from slmpc.envs.pointbot import PointBot
from slmpc.envs.cartpole import CartPole
from slmpc.controllers import LMPC, RandomController
from slmpc.misc import NoSwitchSchedule, SingleSwitchSchedule

def pointbot_config(exp_cfg):
	exp_cfg.save_dir = "logs/pointbot"
	exp_cfg.demo_path = "demos/pointbot/demos.p"
	exp_cfg.ss_approx_mode = "knn" # Should change to 'convex_hull'
	exp_cfg.value_approx_mode = "pe" # could be linear too, but I am pretty sure knn is better
	exp_cfg.variable_start_state = True
	exp_cfg.variable_start_state_cost = "towards" # options are [indicator, nearest_neighbor, towards]
	exp_cfg.soln_mode = "cem"
	exp_cfg.alpha_thresh = 3
	exp_cfg.parallelize_cem = False
	exp_cfg.parallelize_rollouts = True
	exp_cfg.model_logdir = 'model_logs'
	exp_cfg.optimizer_params = {"num_iters": 5, "popsize": 200, "npart": 1, "num_elites": 40, "plan_hor": 15, "per": 1, "alpha": 0.1, "extra_hor": 5} # These kind of work for pointbot?
	exp_cfg.n_samples_start_state_opt = 5
	exp_cfg.start_state_opt_success_thresh = 0.6
	exp_cfg.ss_value_train_success_thresh = 0.7
	exp_cfg.desired_starts = [[-100, 0, 0, 0] for _ in range(exp_cfg.num_iterations)] # Placeholder for now
	exp_cfg.update_SS_and_value_func_CEM = False
	exp_cfg.max_update_SS_value = 50
	return PointBot()

def cartpole_config(exp_cfg):
	exp_cfg.soln_mode = "cem"
	exp_cfg.alpha_thresh = 5
	exp_cfg.parallelize_cem = True
	exp_cfg.parallelize_rollouts = False
	exp_cfg.save_dir = "logs/cartpole"
	exp_cfg.demo_path = "demos/cartpole/demos.p"
	exp_cfg.ss_approx_mode = "knn"
	exp_cfg.variable_start_state = False
	exp_cfg.value_approx_mode = "pe" # could be linear too, but I am pretty sure knn is better
	exp_cfg.model_logdir = 'model_logs'
	exp_cfg.optimizer_params = {"num_iters": 5, "popsize": 200, "npart": 1, "num_elites": 40, "plan_hor": 20, "per": 1, "alpha": 0.1, "extra_hor": 5} # These kind of work for cartpole
	exp_cfg.n_samples_start_state_opt = 5
	exp_cfg.start_state_opt_success_thresh = 0.6
	exp_cfg.ss_value_train_success_thresh = 0.6
	exp_cfg.desired_starts = [[-50, 0, 50, 0] for _ in range(exp_cfg.num_iterations)] # Placeholder for now
	exp_cfg.update_SS_and_value_func_CEM = False
	exp_cfg.max_update_SS_value = 50
	return CartPole()


def pointbot_exp1_config(exp_cfg):
	exp_cfg.samples_per_iteration = 5
	exp_cfg.num_iterations = 15
	from slmpc.envs.pointbot_const import GOAL_STATE
	exp_cfg.goal_schedule = NoSwitchSchedule(None, GOAL_STATE)
	exp_cfg.demo_path = "demos/pointbot/demos_1.p"

def pointbot_exp2_config(exp_cfg):
	exp_cfg.samples_per_iteration = 5
	exp_cfg.num_iterations = 5
	from slmpc.envs.pointbot_const import GOAL_STATE
	exp_cfg.goal_schedule = SingleSwitchSchedule(2, [GOAL_STATE, GOAL_STATE])


def config(env_name, controller_type, exp_id):
	exp_cfg = DotMap()
	exp_cfg.controller_type = controller_type
	exp_cfg.log_all_data = True

	# experiment specific overrides
	if exp_id == 'p1':
		pointbot_exp1_config(exp_cfg)
	elif exp_id == 'p2':
		pointbot_exp2_config(exp_cfg)
	else:
		raise Exception("Unknown Experiment ID.")


	if env_name == "pointbot":
		env = pointbot_config(exp_cfg)
	elif env_name == "cartpole":
		env = cartpole_config(exp_cfg)
	else:
		raise Exception("Unsupported environment.")


	exp_cfg.env_name = env.env_name
	exp_cfg.ac_lb = env.action_space.low
	exp_cfg.ac_ub = env.action_space.high


	return exp_cfg, env

if __name__ == '__main__':
	multiprocessing.set_start_method('spawn')
	parser = argparse.ArgumentParser()
	parser.add_argument('-env_name', type=str, default="pointbot",
						help='Environment name: select from [pointbot, cartpole]')
	parser.add_argument('-ctrl', type=str, default="random",
						help='Controller name: select from [random, lmpc_expect]')
	parser.add_argument('-exp_id', type=str, default="p1",
						help='Experiment ID: select from [p1, p2]')
	args = parser.parse_args()

	exp_cfg, env = config(args.env_name, args.ctrl, args.exp_id)
	experiment = Experiment(env, exp_cfg)
	experiment.run()
	# experiment.plot_results()
	
