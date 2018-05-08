import random
import numpy as np
import dill as pickle
from base_agent import Agent
from collections import defaultdict


class QL(Agent):
    """ A q-learning class based on Víctor Mayoral Vilches' Q-learning algorithm
    from https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial4
    and on the Q-learning (off-policy TD control) algorithm as described in
        Sutton and Barto
        Reinforcement Learning: An Introduction, Chapter 6.5
        2nd edition, Online Draft, January 1 2018 version, retrieved from
        http://incompleteideas.net/book/the-book-2nd.html
    """
    discrete = True

    def __init__(self, env, seed=24, discount=.9, epsilon=.1, lr=.5):
        super().__init__(env, seed)
        self.discount = discount
        self.epsilon = epsilon
        self.lr = lr
        self.q = defaultdict(lambda: 0.)    # key: (state, action), value: corresponding value

    def _select_action(self, state: tuple, eval_mode=False) -> int:
        """
        Get the state-action values and use them as input for selecting a state
        in a epsilon-greedy fashion.

        :param eval_mode:
        :param state:   a state vector
        :return:        an action index
        """
        if random.random() < self.epsilon and not eval_mode:
            return np.random.choice(self._n_actions)
        else:
            q = [self.q[(state, a)] for a in range(self._n_actions)]
            return np.array(q).argmax()

    def _learn_transition(self, state: tuple, action, reward, next_state, done):
        """
        Apply Q-learning rule:
            Q(s, a) += alpha * (reward(s, a) + max(Q(s') - Q(s, a))

        :param state:   an iterable object representing a state
        :param action:  an iterable object representing an action
        :param reward:  a float or int representing the reward received for
                        taking action action in state state
        :param next_state:
        """
        if self.q[(state, action)] == 0.:
            self.q[(state, action)] = reward
        else:
            next_values = [self.q[(next_state, a)] for a in range(self._n_actions)]
            target = reward + self.discount * max(next_values)
            self.q[(state, action)] += self.lr * (target - self.q[(state, action)])

    def save(self, path: str):
        with open(path + '-q.p', 'wb') as f:
            pickle.dump(self.q, f)


class QLP(Agent):
    discrete = True

    def __init__(self, env, seed=24,
                 init_mean=0, init_std=.1,
                 lr_init=.25, lr_decay=.5, decay_freq=500, lr_min=.0001,
                 exploration_start=1, exploration_min=.05, anneal_steps=500,
                 discount=.9,
                 idealization=.75):
        super().__init__(env, seed)
        self.discount = discount
        self.idealization = idealization

        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self.decay_freq = decay_freq
        self._lr = lr_init

        self.exploration_start = exploration_start
        self.exploration_min = exploration_min
        self.anneal_steps = anneal_steps
        self._exploration_eps = exploration_start
        self._exploration_drop = (exploration_start - exploration_min) / anneal_steps

        self.init_mean = init_mean
        self.init_std = init_std
        self.q = defaultdict(lambda: np.random.normal(init_mean, init_std, size=self._n_actions))    # state: value of each action

    @property
    def config(self) -> dict:
        c = super().config
        c.update({attr: getattr(self, attr) for attr in [
            'init_mean', 'init_std',
            'lr_init', 'lr_decay', 'decay_freq', 'lr_min',
            'discount', 'idealization',
            'exploration_start', 'exploration_min', 'anneal_steps',
        ]})
        return c

    def _select_action(self, state: tuple, eval_mode=False) -> int:
        # Explore
        if random.random() < self._exploration_eps and not eval_mode:
            return np.random.choice(self._n_actions)
        # Exploit
        else:
            return self.q[state].argmax()

    def train(self):
        super().train()
        # Perform episode end updates
        self._exploration_eps = max(self._exploration_eps - self._exploration_drop, self.exploration_min)
        if self._episode % self.decay_freq == 0:
            self._lr = max(self._lr * self.lr_decay, self.lr_min)

    def _learn_transition(self, state: tuple, action, reward, next_state, done):
        if done:
            target = reward
        else:
            next_values = self.q[next_state]
            next_avg = next_values.mean()
            next_max = next_values.max()
            next_agg = next_avg + (next_max - next_avg) * self.idealization
            target = reward + self.discount * next_agg  # add future discount only if not done

        self.q[state][action] += self._lr * (target - self.q[state][action])

    def save(self, path: str):
        with open(path + '-q.p', 'wb') as f:
            pickle.dump(self.q, f)
