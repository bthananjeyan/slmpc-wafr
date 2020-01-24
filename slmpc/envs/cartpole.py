from .inverted_pendulum import InvertedPendulum
from scipy.integrate import solve_ivp
import control
import code

import os
import pickle
import os.path as osp
import numpy as np
import cv2

from gym import Env
from gym import utils
from gym.spaces import Box
from .cartpole_const import *
from scipy.stats import truncnorm
import pickle

def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)

def y_dot_action(t, y, u):
    g = 9.8 # Gravitational Acceleration
    L = 1.5 # Length of pendulum
    m = 1.0 # mass of bob (kg)
    M = 5.0 # mass of cart (kg)
    d1 = 1.0
    d2 = 0.5

    x_ddot = u - m*L*y[3]*y[3] * np.cos( y[2] ) + m*g*np.cos(y[2]) *  np.sin(y[2])
    x_ddot = x_ddot / ( M+m-m* np.sin(y[2])* np.sin(y[2]) )
    theta_ddot = -g/L * np.cos( y[2] ) -  np.sin( y[2] ) / L * x_ddot
    damping_x =  - d1*y[1]
    damping_theta =  - d2*y[3]

    return [ y[1], x_ddot + damping_x, y[3], theta_ddot + damping_theta ]

def y_dot_action_vec(t, y, u):
    g = 9.8 # Gravitational Acceleration
    L = 1.5 # Length of pendulum
    m = 1.0 # mass of bob (kg)
    M = 5.0 # mass of cart (kg)
    d1 = 1.0
    d2 = 0.5

    x_ddot = u - m * L * np.multiply(np.multiply(y[:,3], y[:,3]), np.cos(y[:,2])) + m * g * np.multiply(np.cos(y[:,2]), np.sin(y[:,2]))
    x_ddot = np.divide(x_ddot, ( M + m - m * np.multiply(np.sin(y[:,2]), np.sin(y[:,2]))))
    theta_ddot = -g / L * np.cos(y[:,2]) -  np.multiply(np.sin(y[:,2]), x_ddot)/L
    damping_x =  - d1*y[:,1]
    damping_theta =  - d2*y[:,3]

    return np.stack([y[:,1], x_ddot + damping_x, y[:,3], theta_ddot + damping_theta]).T

def runge_kutta_vec(f, y0, tf, steps=100):
    h = tf/steps
    yn = y0
    tn = 0
    for i in range(int(steps)):
        k1 = h[...,np.newaxis] * np.array(f(tn, yn))
        k2 = h[...,np.newaxis] * np.array(f(tn + h/2, yn + k1/2))
        k3 = h[...,np.newaxis] * np.array(f(tn + h/2, yn + k2/2))
        k4 = h[...,np.newaxis] * np.array(f(tn + h, yn + k3))

        yn = yn + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        tn = tn + h
    return yn


y_dot_first = lambda u: (lambda t, y: y_dot_action(t, y, u))
y_dot_first_vec = lambda u: (lambda t, y: y_dot_action_vec(t, y, u))

def test_integrator():
    y0 = np.array([[0, 0, np.pi, 0],[1, 2, np.pi/3, np.pi/8]])
    acs = np.array([1, 2])
    fn = y_dot_first_vec(acs)
    t0 = np.array([0, 0])
    tf = np.array([1, 1])
    oput = runge_kutta_vec(fn, y0, tf)
    print(oput)

    fn = y_dot_first(acs[0])
    sol = solve_ivp(fn, [t0[0], tf[0]], y0[0], t_eval=[tf[0]], rtol=1e-6).y
    print(sol)
    fn = y_dot_first(acs[1])
    sol = solve_ivp(fn, [t0[1], tf[1]], y0[1], t_eval=[tf[1]], rtol=1e-6).y
    print(sol)

class MyLinearizedSystem:
    def __init__(self):
        g = 9.8
        L = 1.5
        m = 1.0
        M = 5.0
        d1 = 1.0
        d2 = 0.5

        # Pendulum up (linearized eq)
        # Eigen val of A : array([[ 1., -0.70710678, -0.07641631,  0.09212131] )
        _q = (m+M) * g / (M*L)
        self.A = np.array([\
                    [0,1,0,0], \
                    [0,-d1, -g*m/M,0],\
                    [0,0,0,1.],\
                    [0,d1/L,_q,-d2] ] )

        self.B = np.expand_dims( np.array( [0, 1.0/M, 0., -1/(M*L)] ) , 1 ) # 4x1

    def compute_K(self, desired_eigs = [-0.1, -0.2, -0.3, -0.4] ):
        # print('[compute_K] desired_eigs=', desired_eigs)
        self.K = control.place( self.A, self.B,  desired_eigs )

    def get_K(self):
        return self.K

class CartPole(Env, utils.EzPickle):
    def __init__(self, cem_env=False):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None
        self.ss = MyLinearizedSystem()
        # Arbitrarily set Eigen Values
        #ss.compute_K(desired_eigs = np.array([-.1, -.2, -.3, -.4])*3. ) # Arbitarily set desired eigen values
        # Eigen Values set by LQR
        self.Q = np.diag( [1,1,1,1.] )
        self.R = np.diag( [1.] )
        # K : State feedback for stability
        # S : Solution to Riccati Equation
        # E : Eigen values of the closed loop system
        self.K, self.S, self.E = control.lqr(self.ss.A, self.ss.B, self.Q, self.R)
        self.ss.compute_K(desired_eigs = self.E) # Arbitarily set desired eigen values
        self.syst = InvertedPendulum()
        self.horizon = HORIZON
        self.action_space = Box(-np.ones(1) * MAX_FORCE, np.ones(1) * MAX_FORCE) # TODO: set this
        self.observation_space = Box(-np.ones(4) * np.float('inf'), np.ones(4) * np.float('inf'))
        self.start_state = START_STATE
        self.goal_state = GOAL_STATE
        self.dt = DT
        self.t = 0
        self.name = "cartpole"
        self.env_name = 'CartPole-v3'
        self.cem_env = cem_env

    def step(self, a):
        a = process_action(a)
        sol = solve_ivp(y_dot_first(a), [self.t, self.t+self.dt], self.state, t_eval=[self.t+self.dt])
        next_state = sol.y[:, -1]
        # next_state += NOISE_SCALE * truncnorm.rvs(-1, 1, size=len(self.state))
        cur_cost = self.step_cost(self.state, a)
        self.cost.append(cur_cost)
        self.state = next_state
        self.time += 1
        self.t += self.dt
        self.hist.append(self.state)
        self.done = HORIZON <= self.time
        if not self.cem_env:
            print("Timestep: ", self.time, " State: ", self.state, " Cost: ", cur_cost)
        return self.state, cur_cost, self.done, {}

    def vectorized_step(self, s, a):
        # fixed start state, execute many sequences of controls in parallel
        state = np.tile(s, (len(a), 1))
        trajectories = [state]
        for t in range(a.shape[1]):
            next_state = runge_kutta_vec(y_dot_first_vec(a[:,t].ravel()), state, np.ones(len(a)) * self.dt)
            # next_state += NOISE_SCALE * truncnorm.rvs(-1, 1, size=next_state.shape)
            trajectories.append(next_state)
            state = next_state
        costs = []
        for t in range(a.shape[1]):
            costs.append(self.step_cost(trajectories[t], a[:,t]))
        return np.stack(trajectories, axis=1), np.array(costs).T

    def reset(self):
        self.state = np.array(self.start_state) # TODO: update this to allow varied start states...
        self.time = 0
        self.cost = []
        self.done = False
        self.hist = [self.state]
        return self.state

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
        return lambda x: (np.abs(s[:,2] - self.goal_state[2]) > GOAL_THRESH).astype(float)

    # TODO: make this not dense cost at some point
    def step_cost(self, s, a):
        if HARD_MODE:
            if len(s.shape) == 2:
                return (np.abs(s[:,2] - self.goal_state[2]) > GOAL_THRESH).astype(float)
            else:
                return float(np.abs(s[2] - self.goal_state[2]) > GOAL_THRESH)
        if len(s.shape) == 2:
            return np.abs(s[:,2] - self.goal_state[2])
        else:
            return np.abs(s[2] - self.goal_state[2])

    def values(self):
        return np.cumsum(np.array(self.cost)[::-1])[::-1]

    def sample(self):
        return np.random.random(1) * 2 * MAX_FORCE - MAX_FORCE

    # Returns whether a state is stable or not
    def is_stable(self, s):
        return np.abs(s[2] - self.goal_state[2]) <= GOAL_THRESH

    # This will be our LQR Controller.
    # LQRs are more theoritically grounded, they are a class of optimal control algorithms.
    # The control law is u = KY. K is the unknown which is computed as a solution to minimization problem.
    def lqr_u(self, y):
        u_ = -np.matmul(self.ss.K , y - np.array(self.goal_state) ) # This was important
        return np.asarray(u_[0])[0]

    def teacher(self, sess=None):
        return CartPoleTeacher()

class CartPoleTeacher(object):
    def __init__(self):
        self.env = CartPole()
        self.demonstrations = []
        self.outdir = "demos/cartpole"

    def get_rollout(self):
        obs = self.env.reset()
        O, A, cost_sum, costs = [obs], [], 0, []
        noise_std = NOISE_STD
        for i in range(HORIZON):
            noise_idx = np.random.randint(int(HORIZON * 3 / 4))
            action = self.env.lqr_u(obs)

            # if i < noise_idx:
            #     action = (np.array(action) +  np.random.normal(0, noise_std, self.env.action_space.shape[0])).tolist()

            A.append(action)

            # test vectorized dynamics
            # o, ac = np.array(obs)[...,np.newaxis].T, np.array(action)[...,np.newaxis,np.newaxis]
            # obs, cost = self.env.vectorized_step(o, ac)
            # obs = obs[0,1]
            # cost = cost[0,0]

            obs, cost, done, _ = self.env.step(action)


            O.append(obs)
            cost_sum += cost
            costs.append(cost)

        values = np.cumsum(costs[::-1])[::-1]
        print("OBS", obs, costs)
        if self.env.is_stable(obs):
            stabilizable_obs = O
        else:
            stabilizable_obs = []
            return self.get_rollout()

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
        pickle.dump(rollouts, open( osp.join(self.outdir, "demos.p"), "wb" ) )

# Both cart and the pendulum can move.
if __name__=="__main__":

    syst = InvertedPendulum()
    env = CartPole()
    obs = env.reset()
    teacher = env.teacher()
    teacher.save_demos(100)
    print("DONE DEMOS")

    rollout = teacher.get_rollout()
    obs_rollout = rollout["obs"]
    acs_rollout = rollout["ac"]
    print("ACS ROLLOUT", acs_rollout)

    for i in range(len(obs_rollout)):
        t = DT * i
        rendered = syst.step(obs_rollout[i], t)
        cv2.imshow( 'im', rendered )
        cv2.moveWindow( 'im', 100, 100 )

        if cv2.waitKey(0) == ord('q'):
            break
