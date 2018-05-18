"""
Contains frame for running and reporting results of multiple experiments in parallel.
Note: seed 7 solves LunarLander in 300 episodes (17 min on latinum server)
  and seed 0 solves in 340 episodes (19 min)
"""

import argparse
from time import time
from multiprocessing import Pool
from typing import Optional
import pandas as pd

from my_agent import MyAgent
from lunarlander_wrapper import LunarLanderWrapper


parser = argparse.ArgumentParser(description="Experiment parameters")
parser.add_argument('-e', '--episodes', type=int, default=500, help="The maximum number of episodes per run")
parser.add_argument('-r', '--runs',     type=int, default=10,  help="The number of runs (repeats) of the experiment")
args = parser.parse_args()


def run(run_number: int) -> ([float], Optional[int], float):
    """ Train agent for multiple episodes on a fresh environment and return
        episode reward history, solve episode (or None if not solved) and time (in seconds) it took. """
    rewards = []  # total reward recorded at the end of each episode
    start_time = time()

    wrapper = LunarLanderWrapper()
    agent = MyAgent(wrapper=wrapper, seed=run_number)

    for episode in range(args.episodes):  # train for the given number of episodes
        rewards.append(agent.train())
        if wrapper.solved(rewards):  # exit early if environment is solved
            break
    else:  # no loop break, means the environment wasn't solved in the given episode budget
        episode = None

    duration = time() - start_time
    return rewards, episode, duration


if __name__ == '__main__':
    """ Run each experiment on its separate thread and save their results as csv files. """
    n_runs = args.runs
    results = Pool(n_runs).map(run, range(n_runs))

    stats_dicts = []
    rewards_dfs = []

    for run_number, (rewards, solve_episode, duration) in enumerate(results):
        stats_dicts.append(dict(run_number=run_number, solve_episode=solve_episode, seconds_taken=duration))
        df = pd.DataFrame(rewards, columns=['reward'])
        df.index.name = 'episode'
        df['run_number'] = run_number
        rewards_dfs.append(df)

    pd.DataFrame(stats_dicts).to_csv('run_stats.csv', index=False)  # final episode, time taken for each run
    pd.concat(rewards_dfs).to_csv('rewards.csv')  # reward per episode for each run
