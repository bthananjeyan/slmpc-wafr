"""
Constants associated with the PointBot env.
"""
import numpy as np 

START_STATE = [0.0, 0., np.pi/4, 0. ]
GOAL_STATE = [0, 0, np.pi/2, 0]
GOAL_THRESH = 0.003
NOISE_STD = 25
MAX_FORCE = 150

MAX_FORCE = 200
HORIZON = 100
DT = 0.2
NOISE_SCALE = 0.05


