from typing import Union, List, Tuple
import numpy as np
import gym

State = Union[List[float], Tuple[int]]

class StateInfo:
    def __init__(self, bins, low, high=None):
        if high is None:
            lim = low
            low, high = -lim, +lim
        self.bins = bins
        self.low = low
        self.high = high


class EnvWrapper(gym.Env):
    """ A wrapper for an environment helps you to specify how you model the
    environment such that it can interface with a general Reinforcement Learning
    agent."""

    normalize = False   # bring into -1; +1 range
    discretize = False  # bins, state is a tuple
    bins = None

    solve_runs = None
    solve_threshold = None

    def __init__(self, env_name: str, state_infos: [StateInfo] = None):
        # TODO silence "gym.spaces.Box autodetected dtype as <class 'numpy.float32'>. Please provide explicit dtype."
        # won't work:
        # logging.disable
        # logging.basicConfig(level=logging.ERROR)
        # logging.getLogger().disabled = True
        # logging.getLogger().propagate = False
        # logging.disable('warning')
        self._env = gym.make(env_name)

        self.n_steps = 0
        self.total_reward = 0

        self._state_infos = state_infos
        if state_infos:
            lows  = np.array([si.low  for si in state_infos])
            highs = np.array([si.high for si in state_infos])
            self.observation_space = gym.spaces.Box(lows, highs, dtype=float)
            """ a list of lists, such that for a state vector (x0, ..., xn),
            the zeroth element of the list contains the list of bins for variable
            x1, the first element of the list contains the list of bins for variable
            x2, and so on.
            """
            self.bins = [np.linspace(si.low, si.high, si.bins, endpoint=False)[1:]
                         for si in state_infos]

            diff = highs - lows
            self._state_scale = 2 / diff
            self._state_shift = -(lows + highs) / diff

    @property
    def n_actions(self) -> int:
        return self._env.action_space.n

    @property
    def state_size(self) -> int:
        return self.observation_space.shape[0]

    @property
    def config(self) -> dict:
        return dict(
            state_infos=[(si.low, si.high, si.bins) for si in self._state_infos]
        )

    def reset(self) -> State:
        self.n_steps = 0
        self.total_reward = 0
        return self._process_state(self._env.reset())

    def step(self, action: int) -> (State, float, bool):
        state, reward, done, info = self._env.step(action)
        self.n_steps += 1
        self.total_reward += reward
        return self._process_state(state), reward, done

    def close(self):
        self._env.close()

    def render(self, *args, **kwargs):
        self._env.render(*args, **kwargs)

    def solved(self, rewards: [float]) -> bool:
        assert self.solve_runs is not None
        assert self.solve_threshold is not None

        rewards = rewards[-100:]
        return len(rewards) == 100 and rewards.mean() > self.solve_threshold

    def _process_state(self, state: State) -> State:
        if self.normalize:
            """
            [0, 1] range: ((x - a) / (b - a))
            [-1, +1] range: (2 * x - b - a) / (b - a)  
            """
            return state * self._state_scale + self._state_shift
        if self.discretize:
            return tuple(int(np.digitize(v, bins))
                         for (v, bins) in zip(state, self.bins))
        return state
