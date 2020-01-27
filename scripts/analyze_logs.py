import os
import pickle
import numpy as np

if __name__ == "__main__":
	# folder_name = "2020-01-24--10:07:17"
	# folder_name = "2020-01-24--10:06:50"
	folder_name = "2020-01-24--13:13:40"
	data_folder = os.path.join("logs/pointbot", folder_name)

	data = pickle.load( open( os.path.join(data_folder, "samples.pkl"), "rb" ) )
	starts = np.array(data['valid_starts'])
	starts = np.array(  [s[0] for s in starts]  )[:25]

	cost_data = [data['samples'][i] for i in range(1, len(starts)+1)]
	all_cost_data = []

	for j in range(25):
		all_cost_data.append([cost_data[j][i]['total_cost'] for i in range(5)])

	starts = starts[4::5]
	all_cost_data = all_cost_data[4::5]

	print("STARTS", starts)
	print("ALL COST DATA", all_cost_data)


	# total_costs = 
