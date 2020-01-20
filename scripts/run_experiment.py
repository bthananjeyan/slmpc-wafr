import argparse
from dotmap import DotMap

from slmpc.misc.experiment import Experiment
from slmpc.envs.pointbot import PointBot
from slmpc.controllers import LMPC, RandomController

def pointbot_config(exp_cfg):
	exp_cfg.env = env = PointBot()
	exp_cfg.save_dir = "logs/pointbot"

def config(env_name, controller_type):
	exp_cfg = DotMap
	exp_cfg.samples_per_iteration = 10
	exp_cfg.num_iterations = 10
	exp_cfg.controller_type = controller_type
	exp_cfg.log_all_data = False
	exp_cfg.env_name = env_name

	if exp_cfg.env_name == "pointbot":
		pointbot_config(exp_cfg)
	else:
		raise Exception("Unsupported environment.")

	if exp_cfg.controller_type == "random":
		exp_cfg.controller = RandomController(exp_cfg.env)
	elif exp_cfg.controller_type == "lmpc_expect":
		exp_cfg.controller = LMPC(exp_cfg.env)
	else:
		raise Exception("Unsupported controller.")

	return exp_cfg

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-env_name', type=str, default="pointbot",
						help='Environment name: select from [pointbot]')
	parser.add_argument('-ctrl', type=str, default="random",
						help='Controller name: select from [random, lmpc]')
	args = parser.parse_args()

	exp_cfg = config(args.env_name, args.ctrl)
	experiment = Experiment(exp_cfg.controller, exp_cfg.env, exp_cfg)
	experiment.run()
	# experiment.plot_results()
	