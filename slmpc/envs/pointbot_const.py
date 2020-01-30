"""
Constants associated with the PointBot env.
"""

START_POS = [-25, 0]
END_POS = [0, 0]
END_POS2 = [-25, 10]
END_POS3 = [-7, 7]
START_STATE = [START_POS[0], 0, START_POS[1], 0]
GOAL_STATE = [END_POS[0], 0, END_POS[1], 0]
GOAL_STATE2 = [END_POS2[0], 0, END_POS2[1], 0]
GOAL_STATE3 = [END_POS3[0], 0, END_POS3[1], 0]
GOAL_THRESH = 1.
# GOAL_THRESH = 7 # FOR MULTI-GOAL

MAX_FORCE = 1
HORIZON = 50
HARD_MODE = True
NOISE_SCALE = 0.05
AIR_RESIST = 0.2

OBSTACLE = [[-20,-100, -6, -100], [-10, 100, 6, 100]]
HAS_OBSTACLE = True
