import argparse
from dotmap import DotMap

from slmpc.misc.experiment import Experiment
from slmpc.envs.pointbot import PointBot
from slmpc.envs.cartpole import CartPole
from slmpc.controllers import LMPC, RandomController

def pointbot_config(exp_cfg):
	exp_cfg.env = PointBot()
	exp_cfg.save_dir = "logs/pointbot"
	exp_cfg.demo_path = "demos/pointbot/demos_1.p"
	exp_cfg.ss_approx_mode = "knn" # Should change to 'convex_hull'
	exp_cfg.value_approx_mode = "pe" # could be linear too, but I am pretty sure knn is better
	exp_cfg.variable_start_state = False
	exp_cfg.cem_env = PointBot(cem_env=True)
	exp_cfg.soln_mode = "cem"
	exp_cfg.alpha_thresh = 3
	exp_cfg.parallelize_cem = False

def cartpole_config(exp_cfg):
	exp_cfg.env = CartPole()
	exp_cfg.cem_env = CartPole(cem_env=True)
	exp_cfg.soln_mode = "cem"
	exp_cfg.alpha_thresh = 5
	exp_cfg.parallelize_cem = True
	exp_cfg.save_dir = "logs/cartpole"
	exp_cfg.demo_path = "demos/cartpole/demos.p"
	exp_cfg.ss_approx_mode = "knn"
	exp_cfg.variable_start_state = False
	exp_cfg.value_approx_mode = "knn" # could be linear too, but I am pretty sure knn is better

def config(env_name, controller_type):
	exp_cfg = DotMap
	exp_cfg.samples_per_iteration = 2
	exp_cfg.num_iterations = 10
	exp_cfg.controller_type = controller_type
	exp_cfg.log_all_data = False
	exp_cfg.env_name = env_name

	if exp_cfg.env_name == "pointbot":
		pointbot_config(exp_cfg)
	elif exp_cfg.env_name == "cartpole":
		cartpole_config(exp_cfg)
	else:
		raise Exception("Unsupported environment.")

	if exp_cfg.controller_type == "random":
		exp_cfg.controller = RandomController(exp_cfg)
	elif exp_cfg.controller_type == "lmpc_expect":
		exp_cfg.controller = LMPC(exp_cfg)
	else:
		raise Exception("Unsupported controller.")

	return exp_cfg

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-env_name', type=str, default="pointbot",
						help='Environment name: select from [pointbot, cartpole]')
	parser.add_argument('-ctrl', type=str, default="random",
						help='Controller name: select from [random, lmpc_expect]')
	args = parser.parse_args()

	exp_cfg = config(args.env_name, args.ctrl)
	experiment = Experiment(exp_cfg.controller, exp_cfg.env, exp_cfg)
	experiment.run()
	# experiment.plot_results()
	