from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize, dump, forest_minimize
from simulation import run_trial, parser

space = [
    Real(.5, 1, name='exploration_start'),
    Real(.001, .5, name='exploration_min'),
    Integer(40, 300, name='exploration_anneal_steps'),
    Categorical(['eps-greedy', 'max-boltzmann'], prior=[.7, .3], name='policy'),

    Real(0.001, .1, 'log-uniform', name='lr_init'),
    Real(.05, .95, name='lr_decay'),
    Integer(10, 300, name='decay_freq'),

    Real(0, 1, name='idealization'),
    Real(.5, 1, name='discount'),
    Categorical([2, 3], prior=[.8, .2], name='history_len'),

    Integer(0, 100, name='target_update_freq'),

    Integer(10, 1000, name='batch_size'),
    Integer(1, 5, name='n_epochs'),

    Categorical([True, False], prior=[.2, .8], name='normalize'),

    Categorical([0, .1, .3, .5], prior=[.85, .05, .05, .05], name='input_dropout'),
    Categorical([0, .25, .5, .75], prior=[.85, .05, .05, .05], name='input_dropout'),
    Categorical([True, False], prior=[.1, .9], name='batch_normalization'),
    Categorical([-1, 0, 16, 32, 64, 128, 192], prior=[.8, .1, .02, .02, .02, .02, .02],
                name='streams_size'),
    Categorical(['mse', 'mae', 'logcosh'], prior=[.6, .3, .1], name='loss'),
    Categorical(['sigmoid', 'tanh', 'hard_sigmoid', 'relu'], prior=[.5, .2, .1, .2],
                name='hidden_activation'),
    Categorical(['linear', 'softmax'], prior=[.65, .35], name='out_activation'),
    Categorical(['random_uniform', 'lecun_uniform', 'he_uniform', 'glorot_uniform',
                 'random_normal', 'lecun_normal', 'he_normal', 'glorot_normal'],
                name='weights_init'),
    Categorical(['adam', 'nadam', 'adamax', 'rmsprop', 'adagrad', 'adadelta'],
                prior=[.7, .1, .05, .05, .05, .05], name='optimizer'),
    Categorical([
        (32,),
        (64,),
        (128,),
        (192,),
        (256,),
        (384,),
        (512,),
        (32, 64),
        (128, 256),
        (192, 384),
        (256, 512),
        (64, 32),
        (128, 64),
        (192, 64),
        (256, 128),
        (256, 192),
        (384, 256),
        (384, 192),
        (512, 384),
        (32, 32),
        (64, 64),
        (128, 128),
        (192, 192),
        (256, 256),
        (384, 384),
        (512, 512),
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (192, 192, 192),
        (256, 256, 256),
        (384, 384, 384),
        (512, 512, 512),
    ], name='layer_sizes'),

    Integer(5000, 200000, name='memory_size'),
    Integer(1, 10000, name='q_clip'),
]

seed = 0
n_jobs = 2
n_random_starts = 5
n_restarts_optimizer = 2

optimizer_type = 'gp'  # gp forest
acq = 'EI'  # EI PI LCB
kappa = 2  # for LCB, high = exploration

@use_named_args(space)
def objective(**params):
    args = parser.parse_args([
        '--env=lander',
        '--agent=dqnp',
        '--episodes=10000',  # as many as there is time for
        '--max-time=1800',  # 30 min
        '--eval-interval=0',  # disable eval
    ])

    try:
        stats = run_trial(seed, args, **params)
        score = stats['tail_avg'] * .75 +\
                stats['final_reward'] * .15 +\
                stats['max_reward'] * .1 +\
               -stats['tail_std'] * .1

        return -score  # so we minimize this
    except Exception as e:
        print(e, str(e), 'on', params)
        return 100000


# from skopt.callbacks import CheckpointSaver
class CheckpointSaver:
    """ https://github.com/scikit-optimize/scikit-optimize/blob/c908bedd842ca538a71698a3b4bb40381a99903b/skopt/callbacks.py """
    def __init__(self, checkpoint_path, **dump_options):
        self.checkpoint_path = checkpoint_path
        self.dump_options = dump_options

    def __call__(self, res):
        dump(res, self.checkpoint_path, **self.dump_options)


if __name__ == '__main__':
    checkpoint_saver = CheckpointSaver("optimizer-ckp", compress=9)

    optimizer_factory = {
        'gp': gp_minimize,
        'forest': forest_minimize,
    }[optimizer_type]

    try:
        optimizer = optimizer_factory(objective, space, random_state=seed,
                             n_restarts_optimizer=n_restarts_optimizer, acq_optimizer='lbfgs',
                             acq_func=acq, n_jobs=n_jobs,
                             n_calls=1000, callback=[checkpoint_saver],  # will run until interrupted
                             )
    except KeyboardInterrupt:
        print('interrupting optimizer')
    dump(optimizer, 'optimizer.p')

# try this kind of parallelization: https://scikit-optimize.github.io/notebooks/parallel-optimization.html
