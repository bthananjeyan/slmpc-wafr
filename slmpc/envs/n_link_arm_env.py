# State is joint angles, actions are delta joint angles

import os
import pickle

import os.path as osp
import numpy as np
from gym import Env
from gym import utils
from gym.spaces import Box
from scipy.stats import truncnorm

from slmpc.controllers.utils import euclidean_goal_fn
from .n_link_arm_env_const import *
import matplotlib.pyplot as plt

def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)

def ang_diff(theta1, theta2):
    """
    Returns the difference between two angles in the range -pi to +pi
    """
    return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi

class NLinkArmEnv(Env, utils.EzPickle):

    def __init__(self, cem_env=False):
        utils.EzPickle.__init__(self)

        self.hist = self.cost = self.done = self.time = self.state = None
        self.horizon = HORIZON
        self.action_space = Box(-np.ones(N_LINKS) * MAX_FORCE, np.ones(N_LINKS) * MAX_FORCE)
        self.observation_space = Box(-np.ones(N_LINKS) * np.float('inf'), np.ones(N_LINKS) * np.float('inf'))
        self.start_state = START_STATE # initial joint angles

        self.name = 'nlinkarm'
        self.env_name = 'NLinkArm-v0'
        self.cem_env = cem_env
        self.link_lengths = np.array([LINK_LENGTH] * N_LINKS)
        self.show_animation = SHOW_ANIMATION

        self.n_links = len(self.link_lengths)

        self.lim = sum(self.link_lengths)
        self.goal_pos = GOAL_POS
        self.goal_state, solution_found = self.inverse_kinematics(self.start_state)

        if not solution_found:
            raise Exception("Invalid goal position")

        if show_animation:  # pragma: no cover
            self.fig = plt.figure()

            plt.ion()
            plt.show()

    # TODO: Vectorize this
    def get_points(self, state):
        points = [[0, 0] for _ in range(self.n_links + 1)]
        for i in range(1, self.n_links + 1):
            points[i][0] = points[i - 1][0] + \
                self.link_lengths[i - 1] * \
                np.cos(np.sum(state[:i]))
            points[i][1] = points[i - 1][1] + \
                self.link_lengths[i - 1] * \
                np.sin(np.sum(state[:i]))

        if show_animation:
            self.plot()

        return points

    def plot(self):  # pragma: no cover
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

        for i in range(self.n_links + 1):
            if i is not self.n_links:
                plt.plot([self.points[i][0], self.points[i + 1][0]],
                         [self.points[i][1], self.points[i + 1][1]], 'r-')
            plt.plot(self.points[i][0], self.points[i][1], 'ko')

        plt.plot(self.goal[0], self.goal[1], 'gx')

        plt.plot([self.end_effector[0], self.goal[0]], [
                 self.end_effector[1], self.goal[1]], 'g--')

        plt.xlim([-self.lim, self.lim])
        plt.ylim([-self.lim, self.lim])
        plt.draw()
        plt.pause(0.0001)


    # Make this vectorizable
    def collision_check(self, state):
        if not CHECK_COLLISIONS:
            return 0
        # Get points on arm
        self.points = self.get_points(state)
        arm_line_seg_list = np.array([ [self.points[i], self.points[j]] for i, j in zip(  range(len(self.points)-1), range(1, len(self.points))) ])
        print("ARM LINE SEG LIST", arm_line_seg_list)
        for line_seg in arm_line_seg_list:
            s1, s2 = line_seg
            t_hat = ((OBSTACLE_CENTER - s1).dot(s2 - s1))/( (s2-s1).dot(s2-s1) )
            t = min(max(t_hat, 0), 1)
            dist = np.linalg.norm(s1 + t*(s1 - s1) - OBSTACLE_CENTER)
            if dist <= OBSTACLE_RADIUS: 
                print("colliding", line_seg)
                print(dist)
                return 1

        return 0

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
        a = process_action(a)
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
        self.state = self.start_state
        self.time = 0
        self.cost = []
        self.done = False
        self.hist = [self.state]
        self.points = get_points(self.state)
        return self.state

    def set_state(self, s):
        self.state = s

    def get_hist(self):
        return self.hist

    def get_costs(self):
        return self.cost

    def set_goal(self, goal_state):
        self.goal_state = goal_state
        self.goal_pos = self.forward_kinematics(self.goal_state)

    @property
    def goal_fn(self):
        return euclidean_goal_fn(self.goal_pos, GOAL_THRESH)

    def _next_state(self, s, a):
        return s + a + NOISE_SCALE * truncnorm.rvs(-1, 1, size=s.shape)

    def step_cost(self, s, a):
        if HARD_MODE:
            if len(s.shape) == 2:
                curr_positions = np.array([self.forward_kinematics(state) for state in s])
                return (np.linalg.norm(np.subtract(self.goal_pos, curr_positions), axis=1) > GOAL_THRESH).astype(float)
            else:
                return (np.linalg.norm(np.subtract(self.goal_pos, self.forward_kinematics(s))) > GOAL_THRESH).astype(float)
        return np.linalg.norm(np.subtract(self.goal_pos, self.forward_kinematics(s)))

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
        return np.linalg.norm(np.subtract(self.goal_pos, self.forward_kinematics(s))) <= GOAL_THRESH

    def inverse_kinematics(self, s, goal_pos=None):
        """
        Calculates the inverse kinematics using the Jacobian inverse method.
        """
        if goal_pos is None:
            goal_pos = self.goal_pos 
        new_state = s
        for iteration in range(N_ITERATIONS):
            errors, distance = self.pos_distance(self.forward_kinematics(new_state), goal_pos)
            if distance < 0.1:
                print("Solution found in %d iterations." % iteration)
                return new_state, True
            J = self.jacobian_inverse(new_state)
            new_state = new_state + np.matmul(J, errors)
        return new_state, False

    # TODO: Vectorize this for step_cost
    def forward_kinematics(self, joint_angles):
        x = y = 0
        for i in range(1, N_LINKS + 1):
            x += self.link_lengths[i - 1] * np.cos(np.sum(joint_angles[:i]))
            y += self.link_lengths[i - 1] * np.sin(np.sum(joint_angles[:i]))
        return np.array([x, y]).T


    def jacobian_inverse(self, joint_angles):
        J = np.zeros((2, N_LINKS))
        for i in range(N_LINKS):
            J[0, i] = 0
            J[1, i] = 0
            for j in range(i, N_LINKS):
                J[0, i] -= self.link_lengths[j] * np.sin(np.sum(joint_angles[:j]))
                J[1, i] += self.link_lengths[j] * np.cos(np.sum(joint_angles[:j]))

        return np.linalg.pinv(J)


    def pos_distance(self, current_pos, new_pos):
        x_diff = new_pos[0] - current_pos[0]
        y_diff = new_pos[1] - current_pos[1]
        return np.array([x_diff, y_diff]).T, np.hypot(x_diff, y_diff)


    def teacher(self, sess=None):
        return NLinkArmEnvTeacher()

class NLinkArmEnvTeacher(object):

    def __init__(self):
        self.env = NLinkArmEnv()
        self.outdir = "demos/nlinkarm"
        self.waypoints = [[2, -3], GOAL_POS]

    def get_rollout(self, start_state=None):
        print("START STATE", start_state)

        obs = self.env.reset()
        if start_state is not None:
            self.env.set_state(np.array(start_state))
            O, A, cost_sum, costs = [np.array(start_state)], [], 0, []
        else:
            O, A, cost_sum, costs = [obs], [], 0, []

        noise_std = 0.02

        waypoint_idx = 0

        for i in range(HORIZON):
            noise_idx = np.random.randint(int(HORIZON * 2 / 3))

            # Calculate waypoint in joint space based on current joint configuration and waypoint in position space
            waypoint_joints, solution_found = self.env.inverse_kinematics(obs, self.waypoints[waypoint_idx])

            if not CHECK_COLLISIONS: 
                action = process_action(KP * ang_diff(self.env.goal_state, obs) * DT)
            else:
                # Plan to correct waypoint
                action =  process_action(KP * ang_diff(waypoint_joints, obs) * DT)

            # Go to next waypoint if reached closed enough
            errors, distance = self.env.pos_distance(  self.env.forward_kinematics(obs), self.waypoints[waypoint_idx] ) 
            if distance < 0.5 and waypoint_idx < len(self.waypoints) - 1:
                waypoint_idx += 1

            if i < HORIZON / 2:
                action = np.array([0.01*np.random.random()] * N_LINKS)

            if i < noise_idx:
                action = process_action((np.array(action) +  np.random.normal(0, noise_std, self.env.action_space.shape[0])).tolist())

            # print("POS", self.env.forward_kinematics(obs))

            A.append(action)
            obs, cost, done, info = self.env.step(action)

            collision = env.collision_check(obs)
            print("COLLISION", collision)
            if collision:
                assert(False)
                return self.get_rollout()

            O.append(obs)
            cost_sum += cost
            costs.append(cost)
            if done:
                break

        values = np.cumsum(costs[::-1])[::-1]
        if self.env.is_stable(obs):
            print("STABLE")
            stabilizable_obs = O
        else:
            stabilizable_obs = []
            return self.get_rollout()

        print("COSTS", costs)
        print("POS", [self.env.forward_kinematics(obs) for obs in O])

        return {
            "obs": O,
            "ac": A,
            "cost_sum": cost_sum,
            "costs": costs,
            "values": values,
            "stabilizable_obs" : stabilizable_obs
        }

    def save_demos(self, num_demos):
        rollouts = [teacher.get_rollout() for i in range(num_demos)]
        pickle.dump(rollouts, open( osp.join(self.outdir, "demos_fake.p"), "wb" ) )

if __name__=="__main__":
    env = NLinkArmEnv()
    teacher = env.teacher()
    teacher.save_demos(1)
    print("FINSHED")

    