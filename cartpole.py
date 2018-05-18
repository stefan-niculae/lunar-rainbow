import math
from env_wrapper import EnvWrapper, StateInfo


class CartPole(EnvWrapper):
    solve_runs = 100
    solve_threshold = 195

    def __init__(self, discretization='observed', penalty=-20):
        if discretization == 'theory':
            state_infos = [
                StateInfo(10, 2.4),  # lateral position
                StateInfo(10, 12 * 2 * math.pi / 360),  # angle
                StateInfo(10, 1),  # lateral velocity
                StateInfo(10, 3.5),  # angular velocity
            ]
        else:  # data-based
            state_infos = [
                StateInfo(4, .5),    # lateral position
                StateInfo(12, 1.5),  # angle
                StateInfo(4, .25),   # lateral velocity
                StateInfo(12, 2.5),  # angular velocity
            ]
        super().__init__(env_name='CartPole-v1', state_infos=state_infos)
        self.penalty = penalty  # for tipping pole too far or cart running out of frame

    def step(self, action: int) -> (str, float, bool):
        state, reward, done = super().step(action)

        if done:
            # punish falling over
            reward = self.penalty

        if self.n_steps >= 200:
            # truncate episodes to 200 max steps
            done = True

        return state, reward, done
