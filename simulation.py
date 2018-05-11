import argparse
import json
import random
import logging
from datetime import datetime
from time import time
import os
import numpy as np
import pandas as pd

from ql import QL, QLP
from dqn import DQN, DQNP
from cartpole import CartPole
from lunarlander import LunarLander


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

parser = argparse.ArgumentParser(description="Experiment parameters")
parser.add_argument('--env',            type=str, default='lander',       choices=['pole', 'lander'],  help='The environment to run in')
parser.add_argument('--agent',          type=str, default='dqnp',    choices=['ql', 'qlp', 'dqn', 'dqnp'],  help='The agent that learns and performs')
parser.add_argument('--n-jobs',         type=int, default=1,           help='Number of parallel seeds to try (-1 for all)')
# parser.add_argument('--discr',          type=str, default='observed',   choices=['theory', 'observed'],  help='Discretization type')

parser.add_argument('--episodes',       type=int, default=400,        help="The maximum number of episodes per run")
parser.add_argument('--max-time',       type=int, default=30*60,        help="Maximum number of seconds a run is allowed to reach the episodes")

parser.add_argument('--no-train-log',  action='store_false',          help='If given, training logs will not be printed')
parser.add_argument('--eval-interval',  type=int, default=50,          help='After how many episodes to evaluate')
parser.add_argument('--n-evals',        type=int, default=10,           help='How many times to evaluate')
parser.add_argument('--output-dir',     type=str, default='outputs',    help='Folder to store result stats (csv)')


def run_trial(seed: int, args, **agent_params) -> dict:
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

    start_time = time()
    # eval_stats: [pd.DataFrame] = []
    # train_stats: [dict] = []
    eval_stats = []
    train_stats = []
    solve_episode = None

    env = env_class()
    agent = agent_class(env, seed=seed, **agent_params)
    logging.info('Env' +    env_class.__name__ +   str(env.config))
    logging.info('Agent' + agent_class.__name__ + str(agent.config))

    """ train """
    for episode in range(1, args.episodes + 1):
        duration = time() - start_time
        if duration > args.max_time:
            break

        agent.train()
        r, s = env.total_reward, env.n_steps
        if not args.no_train_log:
            logging.info('({seed}) ep {episode}: {r:.2f} reward, {s} steps'.format(**locals()))
        train_stats.append(dict(episode=episode, reward=r, steps=s))

        # evaluation
        if args.eval_interval and episode % args.eval_interval == 0:
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
    return save(agent, env, duration, train_stats, eval_stats, solve_episode, args.output_dir)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)

def save(agent, env, duration, train_stats: [dict], eval_stats: [pd.DataFrame]=None, solve_episode=None, output_dir='outputs', tail_len=100) -> dict:
    date_str = datetime.now().strftime('%d.%m %H.%M')
    h = str(hash(agent)) + '-' + str(random.randint(0, 1e20))
    save_dir = '{output_dir}/({agent.seed}) {agent.__class__.__name__} on {env.__class__.__name__} @ {date_str} [{h}]'.format(**locals())
    os.makedirs(save_dir, exist_ok=True)

    train_df = pd.DataFrame(train_stats)
    train_df.to_csv(save_dir + '/train.csv', index=False)

    tail_avg = train_df.reward.tail(tail_len).mean()
    tail_std = train_df.reward.tail(tail_len).std()
    max_reward = train_df.reward.max()
    final_reward = train_df.reward.iloc[-1]
    n_eps = train_df.episode.iloc[-1]

    if eval_stats:
        eval_df  = pd.concat(eval_stats)
        eval_df .to_csv(save_dir + '/evals.csv', index=False)

        max_reward = eval_df.reward.max()
        final_reward = eval_stats[-1].reward.mean()

    run_stats = dict(
            solve_episode=solve_episode,
            time=duration,
            max_reward=max_reward,
            tail_avg=tail_avg,
            tail_std=tail_std,
            final_reward=final_reward,
            n_eps=n_eps,
        )
    with open(save_dir + '/stats.json', 'w') as f:
        json.dump(run_stats, f, cls=NumpyEncoder)
    with open(save_dir + '/agent.json', 'w') as f:
        json.dump(agent.config, f, cls=NumpyEncoder)
    with open(save_dir + '/env.json', 'w') as f:
        json.dump(env.config, f, cls=NumpyEncoder)

    try:
        agent.save(save_dir + '/model')
    except Exception as e:
        print('could not save model', e, str(e))
    print('[{agent.seed}] {save_dir} saved'.format(**locals()))

    return run_stats
