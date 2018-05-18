""" Contains pre-processing done on the LunarLander environment. """

import numpy as np
import gym


class EnvWrapper(gym.Env):
    """ Base environment wrapper class, with reward counters and episode solve check. """

    solve_runs = None  # need this many episode rewards to determine whether the environment is solved
    solve_threshold = None  # need the last episodes to have an average over this threshold

    def __init__(self, env_name: str):
        """ Create Gym environment object and default reward zero. """
        self._env = gym.make(env_name)
        self.total_reward = 0  # will be consulted at the end of an episode

    @property
    def n_actions(self) -> int:
        """ Number of available actions """
        return self._env.action_space.n

    @property
    def state_size(self) -> int:
        """ Number of state features """
        return self.observation_space.shape[0]

    def reset(self) -> [float]:
        """ Resets reward counter, starts a new episode and returns initial state. """
        self.total_reward = 0
        return self._env.reset()

    def step(self, action: int) -> ([float], float, bool):
        """ Take the action given and return resulting state, reward and whether it is terminal. """
        state, reward, done, info = self._env.step(action)
        self.total_reward += reward
        return state, reward, done

    def close(self):
        self._env.close()

    def render(self, *args, **kwargs):
        self._env.render(*args, **kwargs)

    def solved(self, rewards: [float]) -> bool:
        """ Checks whether the latest runs meet the desired reward threshold. """
        if len(rewards) < self.solve_runs:  # not enough runs
            return False
        rewards = rewards[-self.solve_runs:]  # keep only most recent ones
        return np.array(rewards).mean() > self.solve_threshold


class LunarLanderWrapper(EnvWrapper):
    """
    Wrapper over the LunarLander-v2 environment.
    Feeds the current state and previous one, concatenated, for a total of 16 features.

    Actions: main engine | right engine | left engine | do nothing
    Reward: combination of proximity to landing and close to zero speed
    """

    solve_runs = 100  # look at last 100 runs
    solve_threshold = 200  # average over 200 reward

    def __init__(self):
        """ Create the Gym environment and update size of state feature vector.  """
        super().__init__(env_name='LunarLander-v2')

        low  = self._env.observation_space.low
        high = self._env.observation_space.high
        self.observation_space = gym.spaces.Box(np.repeat(low,  2),
                                                np.repeat(high, 2),
                                                dtype=float)
        self.prev_state = None

    def reset(self):
        """ Return the initial state, to which a dummy state feature vector is concatenated. """
        initial_state = super().reset()  # take the original 8 state features
        self.prev_state = initial_state  # store for future step calls
        blank_prev_state = np.zeros(self.state_size // 2)  # because there is no previous state
        return np.concatenate([blank_prev_state, initial_state])

    def step(self, action: int) -> ([float], float, bool):
        """ Keep returned values the same except state, to which the previous one is concatenated. """
        current_state, reward, done = super().step(action)  # take the original 8 state features
        concatenated = np.concatenate([self.prev_state, current_state])  # concat previous 8 ones
        self.prev_state = current_state  # store for future step calls
        return concatenated, reward, done
