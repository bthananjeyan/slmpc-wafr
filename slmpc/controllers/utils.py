import numpy as np

def process_sample_for_goal(sample, goal_fn, indicator_cost=True):
	"""
	Takes in a vectorized indicator function for goal.
	"""
	if not indicator_cost:
		raise NotImplementedError
	goal_reached = goal_fn(sample['states'])
	last_idx = np.max(np.where(goal_reached))
	new_sample = {}
	new_sample['states'] = sample['states'][:last_idx]
	new_sample['costs'] = goal_reached[:last_idx]
	if "actions" in sample:
		new_sample['actions'] = sample['actions'][:last_idx]

	new_sample['total_cost'] = np.sum(new_sample['costs'])
	new_sample['values'] = np.cumsum(new_sample['costs'][::-1])[::-1]

	return new_sample

def euclidean_goal_fn(center, radius):
	return lambda x: np.linalg.norm(x - center, axis=1)

