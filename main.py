import argparse
import os
from datetime import datetime
from multiprocessing import cpu_count, Pool
import pandas as pd
from time import time

from qlearner import QL, QLP
from dqn import DQN, DQNP
from cartpole_wrapper import CartPoleWrapper
from lunarlander_wrapper import LunarLanderWrapper


# Parse the arguments
parser = argparse.ArgumentParser(description="Experiment parameters")
parser.add_argument('--env',            type=str, default='lander',       choices=['pole', 'lander'],  help='The environment to run in')
parser.add_argument('--agent',          type=str, default='dqn',    choices=['ql', 'qlp', 'dqn', 'dqnp'],  help='The agent that learns and performs')
# parser.add_argument('--discr',          type=str, default='observed',   choices=['theory', 'observed'],  help='Discretization type')
parser.add_argument('--episodes',       type=int, default=400,        help="The maximum number of episodes per run")
parser.add_argument('--eval-interval',  type=int, default=10,          help='After how many episodes to evaluate')
parser.add_argument('--n-evals',        type=int, default=3,           help='How many times to evaluate')
parser.add_argument('--n-jobs',         type=int, default=4,           help='Number of parallel seeds to try (-1 for all)')
parser.add_argument('--output-dir',     type=str, default='outputs',    help='Folder to store result stats (csv)')

args = parser.parse_args()

if args.n_jobs == -1:
    args.n_jobs = cpu_count()
env_class = {
    'pole': CartPoleWrapper,
    'lander': LunarLanderWrapper,
}[args.env]
agent_class = {
    'ql': QL,
    'qlp': QLP,
    'dqn': DQN,
    'dqnp': DQNP
}[args.agent]

os.makedirs(args.output_dir, exist_ok=True)
date_str = datetime.now().strftime('%d.%m %H.%M')
out_dir = '{args.output_dir}/{args.agent} on {args.env} at {date_str}'.format(**locals())


def perform_run(seed: int) -> (pd.DataFrame, dict):
    start_time = time()
    progress_stats = []

    env = env_class()
    agent = agent_class(env, seed=seed)

    episode = 0
    for episode in range(1, args.episodes + 1):
        # print('ep', episode, end=' ')
        agent.train()
        # print('reward', env.total_reward, 'n steps', env.n_steps)
        # progress_stats.append({'ep': episode, 'r': env.total_reward, 'steps': env.n_steps})

        if episode % args.eval_interval == 0:
            eval_stats = pd.DataFrame(agent.eval())
            eval_stats['episode'] = episode

            progress_stats.append(eval_stats)

            r = eval_stats.reward
            avg, std, max = r.mean(), r.std(), r.max()
            print('[{seed}] ep {episode}: {avg:.2f} ± {std:.2f} ({max:.2f})'.format(**locals()))

            # if env.solved(eval_stats.reward):
            #     break

    progress_stats = pd.concat(progress_stats)
    progress_stats['seed'] = seed

    duration = time() - start_time
    run_stats = {
        'n_episodes': episode,
        'time': duration,
        'seed': seed,
    }

    try:
        agent.save(out_dir + '-' + str(seed))
    except Exception as e:
        print('could not save model', e, str(e))
    print('[{seed}] done in {episode:,} episodes, {duration:.2f} seconds'.format(**locals()))
    return progress_stats, run_stats


if __name__ == '__main__':
    if args.n_jobs > 1:
        p = Pool(args.n_jobs)
        results = p.map(perform_run, range(args.n_jobs))
        progress, runs = zip(*results)
    else:
        progress, runs = perform_run(0)

    # Store results
    pd.concat(progress)     .to_csv('{filename}-progress.csv'.format(**locals()))
    pd.DataFrame(list(runs)).to_csv('{filename}-runs.csv'.format(**locals()))
    print('Saved in {filename}'.format(**locals()))
