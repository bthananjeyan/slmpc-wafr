import scipy.io as sio 


TARGET = np.array([0.13345871, 0.21923056, -0.10861196])
THRESH = 0.05

data = sio.loadmat("logs.mat")
print(data['observations'].shape)





print(data['actions'].shape)
print(data['rewards'].shape)
print(data['returns'].shape)
# print(data['returns'])
print(data.keys())
assert(False)