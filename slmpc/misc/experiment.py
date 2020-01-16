import numpy as np

class Experiment:

	def __init__(self, controller, env, exp_cfg):
		self.controller = controller
		self.env = env
		self.exp_cfg = exp_cfg

		self.samples_per_iteration = self.exp_cfg.samples_per_iteration
		self.num_iterations = self.exp_cfg.num_iterations

	def reset(self):
		self.all_samples = []
		self.mean_costs = []

	def sample(self):
		data = {
			'states': [],
			'actions': [],
			'costs': []
		}
		obs = self.env.reset()
		data['states'].append(obs)
		done = False
		while not done:
			action = self.controller.act(obs)
			obs, cost, done, _ = self.env.step(action)
			data['states'].append(obs)
			data['actions'].append(action)
			data['costs'].append(cost)
		data['total_cost'] = np.sum(data['costs'])
		return data

	def run(self):
		self.reset()

		for i in range(self.num_iterations):
			print("##### Iteration %d #####"%i)
			samples = [self.sample() for _ in range(self.samples_per_iteration)] # TODO: parallelize
			self.all_samples.append(samples)
			self.controller.train(samples)

			mean_cost = np.mean([s['total_cost'] for s in samples])
			self.mean_costs.append(mean_cost)
			print("Average Cost: %f"%mean_cost)

	@property
	def stats(self):
		return self.mean_costs

	def plot_results(self, save_file=None):
		import matplotlib.pyplot as plt
		plt.plot(self.stats)
		plt.title("Mean Trajectory Cost vs. Iteration")
		plt.xlabel("Iteration")
		plt.ylabel("Trajectory Cost")
		if save_file is not None:
			plt.savefig(save_file)
		plt.show()
