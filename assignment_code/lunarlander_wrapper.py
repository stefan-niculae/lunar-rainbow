from typing import Union, Tuple
import numpy as np
import gym


State = Union[np.array, Tuple[int]]


class EnvWrapper(gym.Env):
    """ A wrapper for an environment helps you to specify how you model the
    environment such that it can interface with a general Reinforcement Learning
    agent."""
    solve_runs = None
    solve_threshold = None

    def __init__(self, env_name: str):
        self._env = gym.make(env_name)

        self.n_steps = 0
        self.total_reward = 0

    @property
    def n_actions(self) -> int:
        return self._env.action_space.n

    @property
    def state_size(self) -> int:
        return self.observation_space.shape[0]

    def reset(self) -> State:
        self.n_steps = 0
        self.total_reward = 0
        return self._env.reset()

    def step(self, action: int) -> (State, float, bool):
        state, reward, done, info = self._env.step(action)
        self.n_steps += 1
        self.total_reward += reward
        return state, reward, done

    def close(self):
        self._env.close()

    def render(self, *args, **kwargs):
        self._env.render(*args, **kwargs)

    def solved(self, rewards: [float]) -> bool:
        assert self.solve_threshold is not None

        if self.solve_runs:
            if len(rewards) < self.solve_runs:
                return False
            rewards = rewards[-self.solve_runs:]
        return rewards.mean() > self.solve_threshold


class LunarLanderWrapper(EnvWrapper):
    """ TODO: Add a description for your wrapper
    Doesn't do much.. DQN needs no discretization

    actions: main engine | right engine | left engine | do nothing
    reward: combination of proximity to landing and close to zero speed
    """
    solve_runs = 100
    solve_threshold = 200

    def __init__(self):
        super().__init__(env_name='LunarLander-v2')

        low  = self._env.observation_space.low
        high = self._env.observation_space.high
        space = gym.spaces.Box(np.repeat(low, 2),
                               np.repeat(high, 2), dtype=float)
        self.observation_space = space
        self.prev_state = np.zeros(self.state_size)

    def reset(self):
        initial_state = super().reset()
        self.prev_state = initial_state
        blank_prev_state = np.zeros(self.state_size // 2)
        return np.concatenate([blank_prev_state, initial_state])

    def step(self, action: int):
        current_state, reward, done = super().step(action)

        concatenated = np.concatenate([self.prev_state, current_state])
        self.prev_state = current_state  # update in memory
        return concatenated, reward, done
