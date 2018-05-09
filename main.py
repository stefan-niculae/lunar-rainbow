import argparse
from multiprocessing import cpu_count, Pool
import pandas as pd
from time import time
import logging

from ql import QL, QLP
from dqn import DQN, DQNP
from cartpole import CartPole
from lunarlander import LunarLander
from simulation import save


parser = argparse.ArgumentParser(description="Experiment parameters")
parser.add_argument('--env',            type=str, default='lander',       choices=['pole', 'lander'],  help='The environment to run in')
parser.add_argument('--agent',          type=str, default='dqnp',    choices=['ql', 'qlp', 'dqn', 'dqnp'],  help='The agent that learns and performs')
# parser.add_argument('--discr',          type=str, default='observed',   choices=['theory', 'observed'],  help='Discretization type')
parser.add_argument('--episodes',       type=int, default=400,        help="The maximum number of episodes per run")
parser.add_argument('--eval-interval',  type=int, default=50,          help='After how many episodes to evaluate')
parser.add_argument('--n-evals',        type=int, default=10,           help='How many times to evaluate')
parser.add_argument('--n-jobs',         type=int, default=1,           help='Number of parallel seeds to try (-1 for all)')
parser.add_argument('--output-dir',     type=str, default='outputs',    help='Folder to store result stats (csv)')

log_train_episodes = False

args = parser.parse_args()

if args.n_jobs == -1:
    args.n_jobs = cpu_count()
env_class = {
    'pole': CartPole,
    'lander': LunarLander,
}[args.env]
agent_class = {
    'ql': QL,
    'qlp': QLP,
    'dqn': DQN,
    'dqnp': DQNP
}[args.agent]

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

configs = [
    # dict(lr_init=0.002),
    # dict(decay_freq=100, lr_decay=0.35),
    # dict(lr_init=0.002, decay_freq=100, lr_decay=0.35),
    # dict(discount=.975),
    # dict(discount=.9),
    # dict(discount=.9, lr_init=0.002, decay_freq=100, lr_decay=0.35),
    # dict(discount=.75),
    # dict(exploration_anneal_steps=75),
    # dict(exploration_anneal_steps=75, lr_init=0.002, decay_freq=100, lr_decay=0.35),
    # dict(exploration_anneal_steps=300),
    # dict(exploration_anneal_steps=300, lr_init=0.002, decay_freq=100, lr_decay=0.35),
    dict(history_len=3),
    dict(history_len=4),
    # dict(layer_sizes=(256, 256)),
    # dict(layer_sizes=(384, 256)),
    # dict(layer_sizes=(256, 384)),
    # dict(layer_sizes=(256, 192)),
    # dict(layer_sizes=(192, 192)),
    dict(prioritize_replay=False),
    dict(prioritize_replay=False, history_len=3),
    dict(prioritize_replay=False, history_len=4),
    dict(prioritize_replay=True, priority_exp=.01),
    dict(prioritize_replay=True, priority_exp=.1),
    dict(prioritize_replay=True, priority_exp=.3),
    dict(prioritize_replay=True, priority_exp=1),
    dict(prioritize_replay=True, priority_exp=2),
    dict(prioritize_replay=True, priority_exp=3),
    dict(prioritize_replay=True, priority_exp=5),
    dict(prioritize_replay=True, priority_exp=10),
    dict(prioritize_replay=True, priority_exp=50),
]


def run_and_save(seed: int):
    start_time = time()
    # eval_stats: [pd.DataFrame] = []
    # train_stats: [dict] = []
    eval_stats = []
    train_stats = []
    solve_episode = None

    env = env_class()
    c = configs[seed % len(configs)]
    agent = agent_class(env, seed=seed, **c)
    print('Env',   env_class.__name__,   str(env.config))
    print('Agent', agent_class.__name__, str(agent.config))

    """ train """
    for episode in range(1, args.episodes + 1):
        duration = time() - start_time
        if duration > 2000:
            break

        agent.train()
        r, s = env.total_reward, env.n_steps
        if log_train_episodes:
            logging.info('({seed}) ep {episode}: {r:.2f} reward, {s} steps'.format(**locals()))
        train_stats.append(dict(episode=episode, reward=r, steps=s))

        # evaluation
        if episode % args.eval_interval == 0:
            stats = pd.DataFrame(agent.eval(n_episodes=args.n_evals))
            stats['episode'] = episode
            eval_stats.append(stats)

            r = stats.reward
            avg, std, max_r = r.mean(), r.std(), r.max()
            logging.info('({seed}) eval ep {episode}: {avg:.2f} ± {std:.2f} (max {max_r:.2f})'.format(**locals()))

            if env.solved(stats.reward):
                solve_episode = episode

    """ save """
    duration = time() - start_time
    save(agent, env, duration, train_stats, eval_stats, solve_episode, args.output_dir)


if __name__ == '__main__':
    if args.n_jobs > 1:
        p = Pool(args.n_jobs)
        p.map(run_and_save, range(args.n_jobs))
    else:
        run_and_save(0)
