import numpy as np

def process_sample_for_goal(sample, goal_fn, indicator_cost=True):
	"""
	Takes in a vectorized indicator function for goal.
	"""
	if not indicator_cost:
		raise NotImplementedError
	goal_reached = goal_fn(sample['states'])
	if np.max(goal_reached):
		last_idx = np.max(np.where(goal_reached))
	else:
		return None
	new_sample = {}
	new_sample['states'] = sample['states'][:last_idx]
	new_sample['costs'] = 1-goal_reached[:last_idx]
	if "actions" in sample:
		new_sample['actions'] = sample['actions'][:last_idx]

	new_sample['total_cost'] = np.sum(new_sample['costs'])
	new_sample['values'] = np.cumsum(new_sample['costs'][::-1])[::-1]

	return new_sample

def euclidean_goal_fn_thresh(center, thresh, preproc=None, name=None):
	center = np.array(center)
	if preproc is None:
		preproc = lambda x: x
	if name == "pendulum":
		return lambda x: euclidean_func(x, goal_state=center, goal_thresh=thresh)
	else:
		return lambda x: (np.linalg.norm(preproc(x) - center, axis=1) < thresh).astype(float)

def euclidean_func(x, goal_state, goal_thresh):
	x = np.array(x)
	angles = x[:, :1]
	first = np.linalg.norm(angles - goal_state[0], axis=1)
	second = np.linalg.norm( (2*np.pi - angles) - goal_state[0], axis=1)
	res = np.minimum(first, second)
	return (res < goal_thresh).astype(float)

