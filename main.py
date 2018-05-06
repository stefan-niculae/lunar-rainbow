import argparse
import os
from datetime import datetime
from multiprocessing import cpu_count, Pool
import pandas as pd
from time import time
import json
import logging

from ql import QL, QLP
from dqn import DQN, DQNP
from cartpole import CartPole
from lunarlander import LunarLander


# Parse the arguments
parser = argparse.ArgumentParser(description="Experiment parameters")
parser.add_argument('--env',            type=str, default='lander',       choices=['pole', 'lander'],  help='The environment to run in')
parser.add_argument('--agent',          type=str, default='dqnp',    choices=['ql', 'qlp', 'dqn', 'dqnp'],  help='The agent that learns and performs')
# parser.add_argument('--discr',          type=str, default='observed',   choices=['theory', 'observed'],  help='Discretization type')
parser.add_argument('--episodes',       type=int, default=400,        help="The maximum number of episodes per run")
parser.add_argument('--eval-interval',  type=int, default=50,          help='After how many episodes to evaluate')
parser.add_argument('--n-evals',        type=int, default=5,           help='How many times to evaluate')
parser.add_argument('--n-jobs',         type=int, default=1,           help='Number of parallel seeds to try (-1 for all)')
parser.add_argument('--output-dir',     type=str, default='outputs',    help='Folder to store result stats (csv)')

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

os.makedirs(args.output_dir, exist_ok=True)
date_str = datetime.now().strftime('%d.%m %H.%M')
out_dir = '{args.output_dir}/{args.agent} on {args.env} @ {date_str}'.format(**locals())

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')


def run_and_save(seed: int):
    start_time = time()
    # eval_stats: [pd.DataFrame] = []
    # train_stats: [dict] = []
    eval_stats = []
    train_stats = []
    solve_episode = None

    env = env_class()
    agent = agent_class(env, seed=seed)
    print('Env',   env_class.__name__,   str(env.config))
    print('Agent', agent_class.__name__, str(agent.config))

    """ train """
    for episode in range(1, args.episodes + 1):
        agent.train()
        r, s = env.total_reward, env.n_steps
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
    save_dir = out_dir + '-' + str(seed)
    os.makedirs(save_dir, exist_ok=True)
    train_df = pd.DataFrame(train_stats)
    eval_df  = pd.concat(eval_stats)
    train_df.to_csv(save_dir + '/train.csv', index=False)
    eval_df .to_csv(save_dir + '/evals.csv', index=False)

    duration = time() - start_time
    with open(save_dir + '/stats.json', 'w') as f:
        json.dump(dict(
            solve_episode=solve_episode,
            time=duration,
            max_reward=max(train_df.reward.max(), eval_df.reward.max()),
        ), f)
    with open(save_dir + '/agent.json', 'w') as f:
        json.dump(agent.config, f)
    with open(save_dir + '/env.json', 'w') as f:
        json.dump(env.config, f)

    try:
        agent.save(save_dir + '/model')
    except Exception as e:
        print('could not save model', e, str(e))
    print('[{seed}] {save_dir} done in {episode:,} episodes, {duration:.2f} seconds'.format(**locals()))


if __name__ == '__main__':
    if args.n_jobs > 1:
        p = Pool(args.n_jobs)
        p.map(run_and_save, range(args.n_jobs))
    else:
        run_and_save(0)