from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import dump, Optimizer, dummy_minimize
from simulation import run_trial, parser
from multiprocessing import Pool
import pandas as pd

space = [
    Real(.8, 1, name='exploration_start'),
    Real(.01, .3, name='exploration_min'),
    Integer(60, 175, name='exploration_anneal_steps'),
    Categorical([-1, .5, 1, 2], prior=[.91, .03, .03, .03], name='exploration_temp'),

    Real(0.001, .05, 'log-uniform', name='lr_init'),
    Real(.05, .25, name='lr_decay'),
    Integer(50, 250, name='decay_freq'),

    Real(0, 1, name='idealization'),
    Real(.9, 1, name='discount'),
    Categorical([2, 3], prior=[.9, .1], name='history_len'),

    Categorical([0, 10, 25, 50], prior=[.85, .05, .05, .05], name='target_update_freq'),

    Categorical([16, 32, 64, 128], prior=[.05, .85, .05, .05], name='batch_size'),
    Integer(1, 2, name='n_epochs'),

    Categorical([True, False], prior=[.1, .9], name='normalize'),

    Categorical([0, .1, .3, .5], prior=[.91, .03, .03, .03], name='input_dropout'),
    Categorical([0, .25, .5, .75], prior=[.91, .03, .03, .03], name='input_dropout'),
    Categorical([True, False], prior=[.1, .9], name='batch_normalization'),
    Categorical([-1, 0, 16, 32, 64, 128, 192], prior=[.88, .02, .02, .02, .02, .02, .02],
                name='streams_size'),
    Categorical(['mse', 'mae', 'logcosh'], prior=[.8, .1, .1], name='loss'),
    Categorical(['sigmoid', 'tanh', 'hard_sigmoid', 'relu'], prior=[.8, .05, .05, .1],
                name='hidden_activation'),
    Categorical(['linear', 'softmax'], prior=[.8, .2], name='out_activation'),
    Categorical(['random_uniform', 'lecun_uniform', 'he_uniform', 'glorot_uniform',
                 'random_normal', 'lecun_normal', 'he_normal', 'glorot_normal'],
                name='weights_init'),
    Categorical(['adam', 'nadam', 'adamax', 'rmsprop', 'adagrad', 'adadelta'],
                prior=[.85, .07, .02, .02, .02, .02], name='optimizer'),
    Categorical([
        # (32,),
        # (64,),
        # (128,),
        # (192,),
        # (256,),
        # (384,),
        (512,),
        # (32, 64),
        # (128, 256),
        (192, 384),
        (256, 512),
        # (64, 32),
        # (128, 64),
        # (192, 64),
        # (256, 128),
        # (256, 192),
        (384, 256),
        (384, 192),
        (512, 384),
        # (32, 32),
        # (64, 64),
        # (128, 128),
        # (192, 192),
        # (256, 256),
        (384, 384),
        # (512, 512),
        # (32, 32, 32),
        # (64, 64, 64),
        # (128, 128, 128),
        (192, 192, 192),
        (256, 256, 256),
        (384, 384, 384),
        # (512, 512, 512),
    ], name='layer_sizes'),

    Categorical([5000, 100000], name='memory_size'),
    Categorical([1, 10, 100, 1000, 10000], name='q_clip'),
]

n_jobs = 8
seed = 0
n_initial_random_samples = 0  # per core


@use_named_args(space)
def score_config(**params):
    args = parser.parse_args([
        '--env=lander',
        '--agent=dqnp',
        '--episodes=1000',  # as many as there is time for
        '--max-time=1500',  # 25 min
        '--eval-interval=0',  # disable eval
    ])

    try:
        score = run_trial(seed, args, **params)
        return -score['aggregated']  # negative because it minimizes
    except Exception as e:
        print(e, str(e), 'on', params)
        return 100000


def get_random_samples(_):
    return dummy_minimize(score_config, space, n_calls=n_initial_random_samples, random_state=seed)


def run_parallel_optimizer(optimizer, save_path='optimizer.p'):
    if n_initial_random_samples > 0:
        rand_results = Pool(n_jobs).map(get_random_samples, range(n_jobs))
        for res in rand_results:
            for x, y in zip(res.x_iters, res.func_vals):
                optimizer.tell(x, y)

    prior_xs = pd.read_csv('prior-xs.csv')[[dim.name for dim in space]]
    prior_xs.layer_sizes = prior_xs.layer_sizes.apply(lambda l: tuple(eval(l)))  # TODO dangerous
    prior_xs.q_clip = prior_xs.q_clip.apply(lambda s: int(s.split()[1][:-1]))
    prior_xs = prior_xs.values

    prior_ys = pd.read_csv('prior-ys.csv').values
    n_out_of_bounds = 0
    for x, y, in zip(prior_xs, prior_ys):
        try:
            optimizer.tell(list(x), float(y[0]))
        except ValueError:
            n_out_of_bounds += 1
    print('out of bounds:', n_out_of_bounds, '/', len(prior_ys))

    while True:  # will run until interrupted
        try:
            xs = optimizer.ask(n_points=n_jobs)      # get suggestion
            ys = Pool(n_jobs).map(score_config, xs)  # report goodness
            opt_result = optimizer.tell(xs, ys)
            dump(opt_result, save_path, compress=9)
        except KeyboardInterrupt:
            print('Stopping hyper-parameter optimization process.')
            break
        except Exception as e:
            print(e, str(e))


if __name__ == '__main__':
    run_parallel_optimizer(
        Optimizer(
            space,
            random_state=seed,
            n_initial_points=0,  # they will be provided in a parallel manner
            base_estimator='GP',  # GP | RF | ET | GBRT
            acq_func='LCB',  # EI | PI | LCB | gp_hedge
            acq_func_kwargs=dict(
                xi=0.01,  # distance between suggestions
                kappa=10,  # if acq is LCB (high = exploration)
            ),
            acq_optimizer='lbfgs',  # sampling | lbfgs
            acq_optimizer_kwargs=dict(
                # n_points=1000,  # if acq optimizer is sampling
                n_restarts_optimizer=5,
                n_jobs=4  # for the optimizer algorithm, not for getting points
            ),
        )
    )
