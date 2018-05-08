import random
from functools import partial
from dqn import DQN, DQNP
from lunarlander import LunarLander
from multiprocessing import Pool
import dill as pickle

from GPyOpt.methods import BayesianOptimization
from simulation import eval_config

env = LunarLander()
agent_class = DQNP

param_space = [
    dict(name='lr_init', type='continuous', domain=(0.001, 0.3)),
    dict(name='lr_decay', type='continuous', domain=(0.05, 0.95)),  # TODO change this in model
    dict(name='decay_freq', type='discrete', domain=list(range(50, 300 + 1, 50))),
    dict(name='discount', type='continuous', domain=(0.1, 1)),
    dict(name='idealization', type='continuous', domain=(0, 1)),
    dict(name='exploration_start', type='continuous', domain=(0.2, 1)),
    dict(name='multi_steps', type='discrete', domain=list(range(1, 5 + 1))),
    dict(name='history_len', type='discrete', domain=list(range(1, 8 + 1))),
    dict(name='layer_sizes', type='discrete', domain=[
        (32,),
        (64,),
        (128,),
        (256,),
        (512,),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (128, 64),
        (256, 128),
        (384, 192),
        (512, 256),
        (128, 128, 128),
        (256, 256, 256),
    ]),
    dict(name='input_dropout', type='continuous', domain=(0, 0.5)),
    dict(name='hidden_dropout', type='continuous', domain=(0, 0.8)),
    dict(name='batch_normalization', type='categorical', domain=[True, False]),
    dict(name='loss', type='categorical', domain=['mse', 'logcosh']),
    dict(name='hidden_activation', type='categorical', domain=['sigmoid', 'tanh', 'relu']),
    dict(name='out_activation', type='categorical', domain=['linear', 'softmax']),
    dict(name='target_update_freq', type='discrete', domain=[20, 50, 100, 150]),
    dict(name='streams_size', type='discrete', domain=[0, 8, 16, 32, 64, 128, 256]),
    dict(name='priority_exp', type='continuous', domain=(0.01, 1.5)),
    dict(name='exploration_anneal_steps', type='discrete', domain=list(range(50, 300 + 1, 50))),
    dict(name='exploration_temp', type='continuous', domain=(0.25, 2.5)),
    dict(name='n_epochs', type='discrete', domain=list(range(1, 3 + 1))),
    dict(name='memory_size', type='discrete', domain=[10000, 25000, 50000, 100000]),
]

n_jobs = 1  # 8
n_init = 1  # 16
max_run_time = 20*60  # in seconds
n_eps = 5  # 300
n_evals = 2  # 20

acq = 'EI'  # EI MPI LCB

def run_and_save(seed):
    random.seed(seed)
    try:
        optimizer = BayesianOptimization(partial(eval_config,
                                                 env=env,
                                                 agent_class=agent_class,
                                                 param_space=param_space,
                                                 max_time=max_run_time,
                                                 n_eps=n_eps,
                                                 n_evals=n_evals,
                                                 agent_seed=seed
                                                 ),
                                         param_space,
                                         initial_design_numdata=n_init, initial_design_type='latin',
                                         acquisition_type=acq,
                                         maximize=True)
        optimizer.run_optimization()
    except KeyboardInterrupt:
        print('interrupted')

    with open('outputs/optimizer-{}[{}].p'.format(acq, seed), 'wb') as f:
        pickle.dump(optimizer, f)


p = Pool(n_jobs)
p.map(run_and_save, range(n_jobs))
