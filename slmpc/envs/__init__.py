from gym.envs.registration import register

register(
	id='PointBot-v0',
	entry_point='slmpc.envs.pointbot:PointBot')

register(
	id='CartPole-v3',
	entry_point='slmpc.envs.cartpole:CartPole')

