import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

N_LINKS = 7
LINK_LENGTH = 1
link_lengths = np.array([LINK_LENGTH] * N_LINKS)

def forward_kinematics(joint_angles):
    if len(joint_angles.shape) == 2:
        x = np.zeros(len(joint_angles))
        y = np.zeros(len(joint_angles))
        for i in range(1, N_LINKS + 1):
            x += link_lengths[i - 1] * np.cos(np.sum(joint_angles[:, :i], axis=1))
            y += link_lengths[i - 1] * np.sin(np.sum(joint_angles[:, :i], axis=1))
        return np.array([x, y]).T
    else:
        x = y = 0
        for i in range(1, N_LINKS + 1):
            x += link_lengths[i - 1] * np.cos(np.sum(joint_angles[:i]))
            y += link_lengths[i - 1] * np.sin(np.sum(joint_angles[:i]))
        return np.array([x, y]).T

def plot_mean_and_CI(mean, ub, lb, color_mean=None, color_shading=None):
	# plot the shaded range of the confidence intervals
	plt.fill_between(range(mean.shape[0]), ub, lb,
					 color=color_shading, alpha=.5)
	# plot the mean on top
	plt.plot(mean, color_mean)

if __name__ == "__main__":
	# OLD: bad runs
	# folder_name = "2020-01-27--13:28:35" # (-50, 0) pointbot (REMOVE, BRIJEN HANDLED THIS)
	# folder_name = "2020-01-27--21:44:00" # (-75, 0) pointbot (REDO)
	# folder_name = "2020-01-27--23:06:05" # (-75, -15) pointbot (REDO)
	# folder_name = "2020-01-27--14:51:56" # (0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3) nlinkarm (CHANGE THIS)

	# Demos from (-25, 0) for pointbot
	# folder_name = "2020-01-27--15:05:06" # (0, 0, 0, 0, 0, 0, 0) nlinkarm fixed start
	# folder_name = "2020-01-28--08:57:36" # (-70, 0) pointbot variable start
	# folder_name = "2020-01-28--11:24:51" # variable start nlinkarm (BAD OLD ONE)
	# folder_name = "2020-01-29--21:23:07" # variable start nlinkarm (NEW, this time planhor=15)
	# folder_name = "2020-01-29--02:36:49" # (-60, -20) pointbot variable start
	folder_name = "2020-02-01--12:31:32"

	# exp_name = "nlinkarm"
	exp_name = "pendulum"
	
	save_file = os.path.join("logs/"+exp_name, folder_name, "costs.png")
	# upper_idx = 10
	upper_idx = 40
	data_folder = os.path.join("logs/"+exp_name, folder_name)
	spacing = 1
	plot = True
	include_demos = True

	data = pickle.load( open( os.path.join(data_folder, "samples.pkl"), "rb" ) )
	starts = np.array(data['valid_starts'])
	starts = np.array([s[0] for s in starts])[:upper_idx]

	cost_data_demos = [data['samples'][0][i]['total_cost'] for i in range(len(data['samples'][0]))]
	print("Demo Cost Mean: ", np.mean(cost_data_demos))
	print("Demo Cost Std: ", np.std(cost_data_demos))

	if include_demos:
		cost_data = [data['samples'][i] for i in range(0, len(starts)+1)]
	else:
		cost_data = [data['samples'][i] for i in range(1, len(starts)+1)]


	constraint_data = [data['samples'][i] for i in range(1, len(starts)+1)]

	all_cost_data = []
	all_constraint_data = []

	for j in range(len(starts)):
		all_cost_data.append([cost_data[j][i]['total_cost'] for i in range(5)])
		all_constraint_data.append([constraint_data[j][i]['collision'] for i in range(5)])

	print("Constraint Violations: ", np.sum(all_constraint_data))

	starts = starts[spacing-1::spacing]
	all_cost_data = all_cost_data[spacing-1::spacing]
	all_constraint_data = all_constraint_data[spacing-1::spacing]
	mean_costs = np.mean(all_cost_data, axis=1)
	std_costs = np.std(all_cost_data, axis=1)
	mean, ub, lb = mean_costs, mean_costs + std_costs, mean_costs - std_costs

	if exp_name == "nlinkarm" and starts[0] is not None:
		print("STARTS", forward_kinematics(starts))
	else:
		print("STARTS", starts)
	print("MEAN COSTS", mean_costs)
	print("STD COSTS", std_costs)
	print("IDXs", np.arange(upper_idx)[spacing-1::spacing] + 1)

	if plot:
		plot_mean_and_CI(mean, ub, lb, 'b', 'b')
		plt.title("N-Link Arm: Mean Trajectory Cost vs. Iteration", fontsize=18)
		plt.xlabel("Iteration", fontsize=18)
		plt.ylabel("Trajectory Cost", fontsize=18)
		plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
		plt.ylim(bottom=0, top=50)
		plt.xlim(left=0, right=10)
		plt.savefig(save_file)
		plt.clf()


