import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class PendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g=10.0):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.horizon = 40
        self.t = 0

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self.state # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        self.t += 1
        return self._get_obs(), -costs, self.t >= self.horizon, {}

    def step_cost(self, o, u):
        state = self._state_from_obs(o)
        th, thdot = state[:,0], state[:,1]
        return np.square(angle_normalize(th)) + 0.1 * np.square(thdot) + 0.001 * np.square(u).ravel()

    def _next_state(self, state, action):
        th, thdot = state[:,0], state[:,1]
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        u = np.clip(action, -self.max_torque, self.max_torque)[0]

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
        state = np.array([np.cos(newth), np.sin(newth), newthdot])
        return state

    def vectorized_step(self, s, a):
        # fixed start state, execute many sequences of controls in parallel
        state = np.tile(s, (len(a), 1))
        trajectories = [state]
        for t in range(a.shape[1]):
            next_state = self._next_state(self._state_from_obs(state), a[:,t].T)
            trajectories.append(self._obs_from_state(next_state.T))
            state = trajectories[-1]
        costs = []
        for t in range(a.shape[1]):
            costs.append(self.step_cost(trajectories[t], a[:,t]))
        return np.stack(trajectories, axis=1), np.array(costs).T


    def reset(self):
        high = np.array([np.pi, 0])
        self.state = self.np_random.uniform(low=high, high=high)
        self.last_u = None
        self.t = 0
        return self._get_obs()

    def is_stable(self, s):
        return angle_normalize(s) < -5*np.pi/8 + angle_normalize(s) > 7 * np.pi

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def _obs_from_state(self, state):
        if len(state.shape) == 1:
            return np.array([np.cos(state[0]), np.sin(state[0]), state[1]])
        return np.stack([np.cos(state[:,0]), np.sin(state[:,0]), state[:,1]]).T

    def _state_from_obs(self, obs):
        if len(obs.shape) == 1:
            theta = np.arctan2(obs[1], obs[0])
            return np.array([theta, obs[2]])
        theta = np.arctan2(obs[:,1], obs[:,0])
        out =  np.stack((theta, obs[:,2])).T
        return out

    def set_goal(self, goal):
        return

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class PendulumTeacher:

    def __init__(self):
        self.env = PendulumEnv()

    def sample(self, render=False):
        costs = []
        actions = []
        o = self.env.reset()
        obs = [o]
        done = False
        while not done:
            u = np.random.randn(1) * 1.5
            o, c, done, _ = self.env.step(u)
            obs.append(o)
            costs.append(c)
            actions.append(u)
            if render:
                self.env.render()

        if self.env.is_stable(obs):
            stabilizable_obs = np.array(o)
        else:
            print('d')
            return self.get_rollout()

        return {
            'states': np.array(o),
            'costs': np.array(costs),
            'actions': np.array(actions),
            'cost_sum': np.sum(costs),
            'values': np.cumsum(np.array(costs)[::-1])[::-1],
            "stabilizable_obs": stabilizable_obs
        }



if __name__ == '__main__':
    t = PendulumTeacher()
    t.sample()
