import math
from env_wrapper import EnvWrapper, StateInfo


class CartPole(EnvWrapper):
    """ This is a wrapper for the CartPole environment as described here:
    https://gym.openai.com/envs/CartPole-v1/

    The discretisation is based on the approach taken by Víctor Mayoral Vilches,
    as explained here:
    https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial4

    The following parameters should remain fixed:
    self._actions   Are defined by the environment
    self._pos_lim   Also defined by environment: space in which cart can move
    self._ang_lim   Also defined by environment: maximum allowed angle for pole

    The following parameters can be changed in parameter tuning:
    self._penalty       Used to penalise state-actions that either cause the
                        cart to move out of the frame or the pole to tip too far

    actions: move left | move right
    reward: +1 for standing up, -20 for falling
    """
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
        # If we reached a terminal state, reward strong penalty for the
        # learning process
        if done:
            reward = self.penalty

        # If we haven't reached the terminal state, but the number of steps
        # runs out, consider this episode done
        if self.n_steps >= 200:
            done = True

        return state, reward, done
