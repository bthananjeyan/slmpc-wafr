from dotmap import DotMap

from slmpc.misc.experiment import Experiment
from slmpc.envs.pointbot import PointBot
from slmpc.controllers.random import RandomController

def config():
	exp_cfg = DotMap
	exp_cfg.samples_per_iteration = 10
	exp_cfg.num_iterations = 10
	exp_cfg.controller_type = "random"
	exp_cfg.env = env = PointBot()

	if exp_cfg.controller_type == "random":
		exp_cfg.controller = RandomController(env)
	else:
		raise Exception("Unsupported controller.")


	return exp_cfg

if __name__ == '__main__':
	exp_cfg = config()
	experiment = Experiment(exp_cfg.controller, exp_cfg.env, exp_cfg)
	experiment.run()
	print(experiment.plot_results())
	