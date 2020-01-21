"""
Constants associated with the PointBot env.
"""
import numpy as np 

START_STATE = [0.0, 0., np.pi/2 + 0.001, 0. ]
GOAL_STATE = [0, 0, np.pi/2, 0]
# GOAL_THRESH = 0.003
GOAL_THRESH = 0.05
NOISE_STD = 5
MAX_FORCE = 10

HORIZON = 100
HARD_MODE = True
DT = 0.2
NOISE_SCALE = 0.05


