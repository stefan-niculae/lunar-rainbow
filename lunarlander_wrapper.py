import numpy as np
import gym
from env_wrapper import EnvWrapper, StateInfo


class LunarLanderWrapper(EnvWrapper):
    """ TODO: Add a description for your wrapper
    Doesn't do much.. DQN needs no discretization

    actions: main engine | right engine | left engine | do nothing
    reward: combination of proximity to landing and close to zero speed
    """
    solve_runs = 100
    solve_threshold = 200

    def __init__(self, discretization='observed', concat_prev_state=True):
        if discretization == 'observed':
            state_infos = [
                StateInfo(10, .75),       # x pos
                StateInfo(10, 0, 1),      # y pos
                StateInfo(5,  1.3),       # x velo
                StateInfo(5,  -1.5, .3),  # y velo
                StateInfo(10, 1.3),       # ang
                StateInfo(5,  .6),        # ang velo
                StateInfo(2, 0, 1),       # left leg (bool)
                StateInfo(2, 0, 1),       # right leg (bool)
            ]
        else:
            raise NotImplementedError
        super().__init__(env_name='LunarLander-v2', state_infos=state_infos)

        # TODO feature selection: don't show last two state values?
        self.concat_prev_state = concat_prev_state
        if self.concat_prev_state:
            low  = self._env.observation_space.low
            high = self._env.observation_space.high
            space = gym.spaces.Box(np.repeat(low, 2),
                                   np.repeat(high, 2))
            self.observation_space = space

            self.prev_state = np.zeros(self.state_size)

        else:
            self.prev_state = None

    def reset(self):
        initial_state = super().reset()
        self.prev_state = initial_state
        blank_prev_state = np.zeros(self.state_size // 2)
        return np.concatenate([blank_prev_state, initial_state])

    def step(self, action: int):
        current_state, reward, done = super().step(action)
        if self.n_actions > 500:  # don't allow it to get stuck
            done = True
            reward -= 100
        if self.concat_prev_state:
            concatenated = np.concatenate([self.prev_state, current_state])
            self.prev_state = current_state  # update in memory
            return concatenated, reward, done
        else:
            return current_state, reward, done
