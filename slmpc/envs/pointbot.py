"""
A robot that can exert force in cardinal directions. The robot's goal is to
reach the origin and it experiences zero-mean Gaussian Noise. State
representation is (x, vx, y, vy). Action representation is (fx, fy),
and mass is assumed to be 1.
"""

import os
import pickle

import os.path as osp
import numpy as np
from gym import Env
from gym import utils
from gym.spaces import Box
from scipy.stats import truncnorm

from slmpc.controllers.utils import euclidean_goal_fn
from .pointbot_const import *

def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)

def lqr_gains(A, B, Q, R, T):
    Ps = [Q]
    Ks = []
    for t in range(T):
        P = Ps[-1]
        Ps.append(Q + A.T.dot(P).dot(A) - A.T.dot(P).dot(B)
            .dot(np.linalg.inv(R + B.T.dot(P).dot(B))).dot(B.T).dot(P).dot(A))
    Ps.reverse()
    for t in range(T):
        Ks.append(-np.linalg.inv(R + B.T.dot(Ps[t+1]).dot(B)).dot(B.T).dot(P).dot(A))
    return Ks, Ps


class PointBot(Env, utils.EzPickle):

    def __init__(self, cem_env=False):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None
        self.A = np.eye(4)
        self.A[2,3] = self.A[0,1] = 1
        self.A[1,1] = self.A[3,3] = 1 - AIR_RESIST
        self.B = np.array([[0,0], [1,0], [0,0], [0,1]])
        self.horizon = HORIZON
        self.action_space = Box(-np.ones(2) * MAX_FORCE, np.ones(2) * MAX_FORCE)
        self.observation_space = Box(-np.ones(4) * np.float('inf'), np.ones(4) * np.float('inf'))
        self.start_state = START_STATE
        self.goal_state = GOAL_STATE
        self.name = "pointbot"
        self.env_name = 'PointBot-v0'
        self.cem_env = cem_env
        self.obstacle = OBSTACLE
        self.has_obstacle = HAS_OBSTACLE


    def step(self, a, log=False):
        a = process_action(a)
        next_state = self._next_state(self.state, a)
        cur_cost = self.step_cost(self.state, a)
        self.cost.append(cur_cost)
        self.state = next_state
        self.time += 1
        self.hist.append(self.state)
        self.done = HORIZON <= self.time
        if not self.cem_env and log:
            print("Timestep: ", self.time, " State: ", self.state, " Cost: ", cur_cost)
        return self.state, cur_cost, self.done, {}

    def vectorized_step(self, s, a):
        # fixed start state, execute many sequences of controls in parallel
        state = np.tile(s, (len(a), 1)).T
        trajectories = [state]
        for t in range(a.shape[1]):
            next_state = self._next_state(state, a[:,t].T)
            trajectories.append(next_state)
            state = next_state
        costs = []
        for t in range(a.shape[1]):
            costs.append(self.step_cost(trajectories[t].T, a[:,t]))
        return np.stack(trajectories, axis=1).T, np.array(costs).T

    def reset(self):
        self.state = self.start_state + np.random.randn(4)
        self.time = 0
        self.cost = []
        self.done = False
        self.hist = [self.state]
        return self.state

    def collision_check(self, state):
        if len(state.shape) == 1:
            state = state[np.newaxis,...]
        if not self.has_obstacle:
            return np.zeros(len(state))
        else:
            below = np.max(np.array(self.obstacle[0]) > state, axis=1)
            above = np.max(state > np.array(self.obstacle[1]), axis=1)
            return np.logical_and(1-below, 1-above)

    def set_state(self, s):
        self.state = s

    def get_hist(self):
        return self.hist

    def get_costs(self):
        return self.cost

    def set_goal(self, goal_state):
        self.goal_state = goal_state

    @property
    def goal_fn(self):
        return euclidean_goal_fn(self.goal_state, GOAL_THRESH)

    def _next_state(self, s, a):
        collisions = self.collision_check(s.T)
        next_state = self.A.dot(s) + self.B.dot(a) + NOISE_SCALE * truncnorm.rvs(-1, 1, size=s.shape)
        return np.where(collisions, s, next_state)

    def step_cost(self, s, a):
        if HARD_MODE:
            if len(s.shape) == 2:
                return (np.linalg.norm(np.subtract(self.goal_state, s), axis=1) > GOAL_THRESH).astype(float)
            else:
                return (np.linalg.norm(np.subtract(self.goal_state, s)) > GOAL_THRESH).astype(float)
        return np.linalg.norm(np.subtract(self.goal_state, s))

    def values(self):
        return np.cumsum(np.array(self.cost)[::-1])[::-1]

    def sample(self):
        return np.random.random(2) * 2 * MAX_FORCE - MAX_FORCE

    def plot_trajectory(self, states=None):
        if states == None:
            states = self.hist
        states = np.array(states)
        plt.scatter(states[:,0], states[:,2])
        plt.show()

    # Returns whether a state is stable or not
    def is_stable(self, s):
        return np.linalg.norm(np.subtract(self.goal_state, s)) <= GOAL_THRESH

    def teacher(self, sess=None):
        return PointBotTeacher()

class PointBotTeacher(object):

    def __init__(self):
        self.env = PointBot()
        self.Ks, self.Ps = lqr_gains(self.env.A, self.env.B, np.eye(4), 50 * np.eye(2), HORIZON)
        self.demonstrations = []
        self.outdir = "demos/pointbot"

    def get_rollout(self, start_state=None):
        print("START STATE", start_state)

        obs = self.env.reset()
        if start_state is not None:
            self.env.set_state(np.array(start_state))
            O, A, cost_sum, costs = [np.array(start_state)], [], 0, []
        else:
            O, A, cost_sum, costs = [obs], [], 0, []

        noise_std = 0.2
        for i in range(HORIZON):
            noise_idx = np.random.randint(int(HORIZON * 2 / 3))
            if i < HORIZON / 3:
                action = [0.2, 0.15]
            elif i < 2 * HORIZON / 3:
                action = [0.2, 0.]
            else:
                action = self._expert_control(obs, i)
            if i < noise_idx:
                action = (np.array(action) +  np.random.normal(0, noise_std, self.env.action_space.shape[0])).tolist()

            action = process_action(action)
            A.append(action)
            obs, cost, done, info = self.env.step(action)
            O.append(obs)
            cost_sum += cost
            costs.append(cost)
            if done:
                break

        values = np.cumsum(costs[::-1])[::-1]
        if self.env.is_stable(obs):
            stabilizable_obs = O
        else:
            stabilizable_obs = []
            return self.get_rollout()

        print(costs)
        # print(O)

        return {
            "obs": O,
            "ac": A,
            "cost_sum": cost_sum,
            "costs": costs,
            "values": values,
            "stabilizable_obs" : stabilizable_obs
        }

    def save_demos(self, num_demos):
        rollouts = [self.get_rollout() for i in range(num_demos)]
        pickle.dump(rollouts, open( osp.join(self.outdir, "demos.p"), "wb" ) )

    def _get_gain(self, t):
        return self.Ks[t]

    def _expert_control(self, s, t):
        return self._get_gain(t).dot(s)

if __name__=="__main__":
    env = PointBot()
    obs = env.reset()
    teacher = env.teacher()
    teacher.save_demos(100)
    print("DONE DEMOS")