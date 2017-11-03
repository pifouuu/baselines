import numpy as np
import math

class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)

class ReplayBuffer(object):
    def __init__(self, limit, content_shape):
        self.contents = {}
        for content,shape in content_shape.items():
            self.contents[content] = RingBuffer(limit, shape=shape)

    def append(self, buffer_item):
        for name, value in self.contents.items():
            value.append(buffer_item[name])

class BaseMemory(object):
    def __init__(self, limit, content_shape):
        self.limit = limit
        self.buffer = ReplayBuffer(limit, content_shape)
        self.observation_shape = content_shape['state0']
        self.action_shape = content_shape['action']

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)
        result = {}
        for name, value in self.buffer.contents.items():
            result[name]=array_min2d(value.get_batch(batch_idxs))

        return result

    def append(self, buffer_item, training=True):
        if not training:
            return
        self.buffer.append(buffer_item)

    def flush(self):
        pass

    def compute_reward(self, state0_batch, action_batch, state1_batch):
        pass

    @property
    def nb_entries(self):
        return len(self.buffer.contents['state0'])

class GoalContinuousMCWrapper(object):
    def __init__(self):
        # Specific to continuous mountain car
        self.obs_to_goal = [0]
        self.state_to_obs = [0,1]
        self.state_to_goal = [2]
        self.state_shape = (3,)
        self.action_shape = (1,)

    def process_state(self, observation, goal):
        return np.concatenate(observation,goal)

    def evaluate_transition(self, state0, action, state1):
        r = 0
        term = False
        if state1[self.state_to_obs][self.obs_to_goal] >= 0.45:
            r += 100
            term = True
        r -= math.pow(action[0], 2) * 0.1
        return r, term

    def sample_goal(self):
        #g = np.random.uniform([-1.2], [0.6], (1,))
        g = [0.45]
        return g


class ContinuousMCWrapper(object):
    def __init__(self):
        # Specific to continuous mountain car
        self.state_shape = (2,)
        self.action_shape = (1,)
        self.obs_to_goal = [0]

    def process_state(self, observation, goal):
        return observation

    def evaluate_transition(self, state0, action, state1):
        r = 0
        term = False
        if state1[self.obs_to_goal] >= 0.45:
            r += 100
            term = True
        r -= math.pow(action[0], 2) * 0.1
        return r, term

    def sample_goal(self):
        # g = np.random.uniform([-1.2], [0.6], (1,))
        g = [0.45]
        return g

class StandardMemory(BaseMemory):
    def __init__(self, env_wrapper, limit):
        self.env_wrapper = env_wrapper
        self.state_shape = self.env_wrapper.state_shape
        self.action_shape = self.env_wrapper.action_shape
        BaseMemory.__init__(self, limit,
                        {'state0': self.state_shape,
                         'action': self.action_shape,
                         'state1':self.state_shape,
                         'reward':(1,),
                         'terminal1':(1,)})


class NoRewardMemory(BaseMemory):
    def __init__(self, env_wrapper, limit):
        # Specific to mountain car continuous
        self.env_wrapper = env_wrapper
        self.state_shape = self.env_wrapper.state_shape
        self.action_shape = self.env_wrapper.action_shape

        BaseMemory.__init__(self, limit,
                        {'state0': self.state_shape,
                         'action': self.action_shape,
                         'state1':self.state_shape})



    def compute_reward(self, state0_batch, action_batch, state1_batch):
        batch_size = state0_batch.shape[0]
        rewards = []
        terminals = []
        for idx in range(batch_size):
            r, term = self.env_wrapper.evaluate_transition(state0_batch[idx],
                                                             action_batch[idx],
                                                             state1_batch[idx])
            rewards.append(r)
            terminals.append(term)
        return array_min2d(rewards), array_min2d(terminals)

    def sample(self, batch_size):
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)
        result = {}
        for name, value in self.buffer.contents:
            result[name] = array_min2d(value.get_batch(batch_idxs))

        result['rewards'], result['terminals1'] = self.compute_reward(result['state0'],
                                                            result['action'],
                                                            result['state1'])
        return result


class HerMemory(StandardMemory):
    def __init__(self, env_wrapper, limit, strategy):
        """Replay buffer that does Hindsight Experience Replay
        obs_to_goal is a function that converts observations to goals
        goal_slice is a slice of indices of goal in observation
        """
        StandardMemory.__init__(self, limit, env_wrapper)

        self.strategy = strategy
        self.data = [] # stores current episode

    def flush(self):
        """Dump the current data into the replay buffer with (final) HER"""
        if not self.data:
            return

        for buffer_item in self.data:
            super().append(buffer_item)
        if self.strategy=='last':
            state_to_goal = self.env_wrapper.state_to_goal
            state_to_obs = self.env_wrapper.state_to_obs
            obs_to_goal = self.env_wrapper.obs_to_goal
            final_state = self.data[-1]['state1']
            new_goal = final_state[state_to_obs][obs_to_goal]
            for buffer_item in self.data:
                buffer_item['state0'][state_to_goal] = new_goal
                buffer_item['state1'][state_to_goal] = new_goal
                buffer_item['reward'], buffer_item['terminal'] = \
                    self.env_wrapper.evaluate_transition(buffer_item['state0'],
                                                           buffer_item['action'],
                                                           buffer_item['state1'])
                super().append(buffer_item)
        else:
            print('error her strategy')
            return
        self.data = []

    def append(self, buffer_item, training=True):
        if not training:
            return
        self.data.append(buffer_item)

    @property
    def nb_entries(self):
        return len(self.buffer.contents['state0'])