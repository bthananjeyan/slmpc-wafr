from random import random
import numpy as np

N_LINKS = 7
LINK_LENGTH = 1
SHOW_ANIMATION = False
N_ITERATIONS = 10000
KP = 2
DT = 0.1

START_STATE = np.array([0] * N_LINKS)
# GOAL_POS = [2, 4]
GOAL_POS = [3, -3]
# GOAL_STATE = [15 * random() - 15 / 2.0,
#             15 * random() - 15 / 2.0]
GOAL_THRESH = 0.5
MAX_FORCE = 0.15
HORIZON = 50
HARD_MODE = True
NOISE_SCALE = 0.03

CHECK_COLLISIONS = False
# # only support circular obstacles for now
OBSTACLE_CENTER = [5.5, -2]
OBSTACLE_RADIUS = 1


