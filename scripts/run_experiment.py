import argparse
from dotmap import DotMap
import multiprocessing

from slmpc.misc.experiment import Experiment
from slmpc.envs.pointbot import PointBot
from slmpc.envs.cartpole import CartPole
from slmpc.controllers import LMPC, RandomController

def pointbot_config(exp_cfg):
	exp_cfg.save_dir = "logs/pointbot"
	exp_cfg.demo_path = "demos/pointbot/demos_1.p"
	exp_cfg.ss_approx_mode = "knn" # Should change to 'convex_hull' since this is a linear system
	exp_cfg.value_approx_mode = "pe"
	exp_cfg.variable_start_state = True
	exp_cfg.variable_start_state_cost = "towards" # options are [indicator, nearest_neighbor, towards]
	exp_cfg.soln_mode = "cem"
	exp_cfg.alpha_thresh = 3
	exp_cfg.parallelize_cem = False
	exp_cfg.parallelize_rollouts = False
	exp_cfg.model_logdir = 'model_logs'
	exp_cfg.optimizer_params = {"num_iters": 5, "popsize": 200, "npart": 1, "num_elites": 40, "plan_hor": 15, "per": 1, "alpha": 0.1, "extra_hor": -5} # These kind of work for pointbot?
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
	exp_cfg.value_approx_mode = "pe"
	exp_cfg.model_logdir = 'model_logs'
	exp_cfg.optimizer_params = {"num_iters": 5, "popsize": 200, "npart": 1, "num_elites": 40, "plan_hor": 20, "per": 1, "alpha": 0.1, "extra_hor": 5} # These kind of work for cartpole
	return CartPole()

def config(env_name, controller_type):
	exp_cfg = DotMap()
	exp_cfg.samples_per_iteration = 2
	exp_cfg.num_iterations = 10
	exp_cfg.n_samples_start_state_opt = 5
	exp_cfg.start_state_opt_success_thresh = 0.6
	exp_cfg.ss_value_train_success_thresh = 0.6
	exp_cfg.controller_type = controller_type
	exp_cfg.log_all_data = False
	exp_cfg.desired_starts = [[-75, 0, 0, 0] for _ in range(exp_cfg.num_iterations)] # Placeholder for now
	exp_cfg.update_SS_and_value_func_CEM = True

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
	args = parser.parse_args()

	exp_cfg, env = config(args.env_name, args.ctrl)
	experiment = Experiment(env, exp_cfg)
	experiment.run()
	# experiment.plot_results()
	