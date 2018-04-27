import random
import numpy as np
from env_wrapper import EnvWrapper, State


class Agent:
    """ Basic agent with some basic functions implemented, such as
    and epsilon-greedy action selection.

    self._env   A subclass of the EnvWrapper class, which translates the
                    environment to an interface for generic Reinforcement
                    Learning Agents
    self._total_reward  Total reward for one training episode

    Also has some basic algorithm parameters:
    self._epsilon   Value in [0, 1] for creating randomness in the greedy method
    self._alpha     Step size parameter

    """
    normalize = False
    discrete = False

    def __init__(self, env: EnvWrapper, seed=42):
        self._env = env  # environment wrapper that provides extra info
        self._state_size = self._env.state_size
        self._n_actions  = self._env.n_actions

        if self.discrete:
            self._env.discretize = True
        if self.normalize:
            self._env.normalize = True

        self._episode = 0

        # set random seed
        random.seed(seed)
        np.random.seed(seed)
        self._env._env.seed(seed)

    def initialise_episode(self):
        """
        Reset the environment through the wrapper.
        :return: initial state vector
        """
        return self._env.reset()

    def train(self):
        """
        Run one episode.
        """
        # Initialise the episode and environment
        state = self.initialise_episode()

        # Main loop of training session
        done = False
        while not done:
            # Select an action, take it and observe outcome
            action = self._select_action(state)
            next_state, reward, done = self._env.step(action)
            self._learn_transition(state, action, reward, next_state, done)
            state = next_state

        self._episode += 1

    def eval(self, n_episodes=100) -> [dict]:
        stats = []
        for episode in range(n_episodes):
            state = self.initialise_episode()
            done = False
            while not done:
                action = self._select_action(state, eval_mode=True)
                state, reward, done = self._env.step(action)
            stats.append({
                'reward': self._env.total_reward,
                'steps':  self._env.n_steps,
            })
        return stats

    def _select_action(self, state: State, eval_mode=False) -> int:
        raise NotImplementedError

    def _learn_transition(self, state: State, action: int, reward: float, next_state: State, done: bool):
        """
        :param state:   an iterable object representing a state
        :param action:  an iterable object representing an action
        :param reward:  a float or int representing the reward received for
                        taking action action in state state
        :param next_state:
        """
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError
