from env_wrapper import EnvWrapper, StateInfo


class LunarLander(EnvWrapper):
    """ TODO: Add a description for your wrapper
    Doesn't do much.. DQN needs no discretization

    actions: main engine | right engine | left engine | do nothing
    reward: combination of proximity to landing and close to zero speed
    """
    # solve_runs = 100
    solve_threshold = 200

    def __init__(self, discretization='observed'):
        if discretization == 'observed':
            state_infos = [
                StateInfo(10, .65),       # x pos
                StateInfo(10, 0, 1),      # y pos
                StateInfo(5,  1),         # x velo
                StateInfo(5,  -1.5, .1),  # y velo
                StateInfo(10, 1),         # ang
                StateInfo(5,  .3),        # ang velo
                StateInfo(2, 0, 1),       # left leg (bool)
                StateInfo(2, 0, 1),       # right leg (bool)
            ]
        else:
            raise NotImplementedError
        super().__init__(env_name='LunarLander-v2', state_infos=state_infos)

        # TODO feature selection: don't show last two state values?

    def step(self, action: int):
        state, reward, done = super().step(action)
        if self.n_actions > 500:  # don't allow it to get stuck
            done = True
            reward -= 100
        return state, reward, done
