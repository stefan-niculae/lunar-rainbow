import random
import numpy as np
from env_wrapper import EnvWrapper, State


class Agent:
    """ Base agent class, with train and eval functions implemented """
    normalize = False
    discrete = False

    def __init__(self, env: EnvWrapper, seed=24):
        """ Set reference variables and random seed """
        self._env = env  # environment wrapper that provides extra info
        self._state_size = self._env.state_size
        self._n_actions  = self._env.n_actions

        if self.discrete:
            self._env.discretize = True
        if self.normalize:
            self._env.normalize = True

        self._episode = 0

        # set random seed
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self._env._env.seed(seed)

    @property
    def config(self) -> dict:
        return dict(
            seed=self.seed
        )

    def initialise_episode(self) -> [float]:
        """
        Reset the environment through the wrapper.
        :return: initial state vector
        """
        return self._env.reset()

    def train(self):
        """ Reset the environment and run for the entire episode until it ends. """
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
        """ Runs the agent multiple times and reports stats received """
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
        """ What action to take in this situation? """
        raise NotImplementedError

    def _learn_transition(self, state: State, action: int, reward: float, next_state: State, done: bool):
        """ Turn this experience into knowledge of how to act in the future. """
        raise NotImplementedError

    def save(self, path: str):
        """ Serialize information about the model so it may be loaded later """
        raise NotImplementedError
