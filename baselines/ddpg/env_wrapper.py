import numpy as np
import math

class GoalContinuousMCWrapper(object):
    def __init__(self):
        # Specific to continuous mountain car
        self.obs_to_goal = [0]
        self.state_to_obs = [0,1]
        self.state_to_goal = [2]
        self.state_shape = (3,)
        self.action_shape = (1,)
        self.eps = 0.2

    def process_state(self, observation, goal):
        return np.concatenate([observation,goal])

    def evaluate_transition(self, state0, action, state1):
        r = 0
        term = False
        if np.abs(state1[self.state_to_obs][self.obs_to_goal] - state1[self.state_to_goal]) < self.eps:
            r += 100
            term = True
        r -= math.pow(action[0], 2) * 0.1
        return r, term

    def sample_goal(self, obs):
        goal_found = False
        while not goal_found:
            goal = np.random.uniform([-1.2], [0.6], (1,))
            goal_found = (obs[self.obs_to_goal]-goal) > self.eps
        return goal

class ContinuousMCWrapper(object):
    def __init__(self):
        # Specific to continuous mountain car
        self.state_shape = (2,)
        self.action_shape = (1,)
        self.reward_shape = (1,)
        self.terminal_shape = (1,)
        self.obs_to_goal = [0]

    def process_observation(self, observation, state):
        return observation

    def evaluate_transition(self, state0, action, state1):
        r = 0
        term = False
        if state1[self.obs_to_goal] >= 0.45:
            r += 100
            term = True
        r -= math.pow(action[0], 2) * 0.1
        return r, term

    def evaluate_goal(self, state):
        return True

    def sample_goal(self, obs):
        return [0.45]

    def process_step(self, state0, action, new_obs, r_env, done_env, info):
        # Compute next complete state
        state1 = self.process_observation(new_obs, state0)

        # Compute reward and terminal condition
        r, done = self.evaluate_transition(state0, action, state1)

        buffer_item = {'state0': state0,
                       'action': action,
                       'reward': r,
                       'state1': state1,
                       'terminal1': done}

        return buffer_item