import argparse
import pandas as pd
from time import time
from multiprocessing import Pool

from my_agent import MyAgent
from lunarlander_wrapper import LunarLanderWrapper

parser = argparse.ArgumentParser(description="Experiment parameters")
parser.add_argument('-e', '--episodes', type=int, default=20,
                    help="The maximum number of episodes per run")
parser.add_argument('-r', '--runs', type=int, default=2,
                    help="The number of runs (repeats) of the experiment")
args = parser.parse_args()
n_episodes = args.episodes
n_runs = args.runs


def run(run_number: int) -> ([float], int, float):
    rewards = []
    start_time = time()

    wrapper = LunarLanderWrapper()
    agent = MyAgent(wrapper=wrapper, seed=run_number)

    for episode in range(n_episodes):
        rewards.append(agent.train())
        if wrapper.solved(rewards):
            break

    wrapper.close()
    duration = time() - start_time

    print('Run {} finished after {} episodes ({:.1f} seconds).'.format(
        run_number, episode, duration))
    return rewards, episode, duration


if __name__ == '__main__':
    results = Pool(n_runs).map(run, range(n_runs))

    stats_dicts = []
    rewards_dfs = []

    for run_number, (rewards, final_episode, duration) in enumerate(results):
        stats_dicts.append(dict(final_episode=final_episode, run_time=duration, run_number=run_number))
        df = pd.DataFrame(rewards, columns=['reward'])
        df.index.name = 'episode'
        df['run_number'] = run_number
        rewards_dfs.append(df)

    pd.DataFrame(stats_dicts).to_csv('run_stats.csv', index=False)
    pd.concat(rewards_dfs).to_csv('rewards.csv')
