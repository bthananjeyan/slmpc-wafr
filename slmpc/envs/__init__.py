from gym.envs.registration import register

register(
	id='PointBot-v0',
	entry_point='slmpc.envs.pointbot:PointBot')

register(
	id='CartPole-v3',
	entry_point='slmpc.envs.cartpole:CartPole')

register(
	id='NLinkArm-v0',
	entry_point='slmpc.envs.n_link_arm_env:NLinkArmEnv')

register(
	id='ReacherSparse-v0',
	entry_point='slmpc.envs.reacher:ReacherSparse3DEnv')