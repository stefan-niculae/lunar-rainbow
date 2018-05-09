import json
from datetime import datetime
from time import time
import os
import pandas as pd


def eval_config(config: dict, env, agent_class, n_eps=300, n_evals=20, max_time=None, param_space=None, agent_seed=24):
    start_time = time()

    if param_space:  # config comes as a list, without names
        config = {p['name']: x for p, x in zip(param_space, config[0])}

    print('\nconfig', config)
    agent = agent_class(env, seed=agent_seed, **config)

    train_stats = []
    for episode in range(n_eps):
        agent.train()
        r, s = env.total_reward, env.n_steps
        train_stats.append(dict(episode=episode, reward=r, steps=s))

        if max_time and time() - start_time > max_time:
            # time limit reached
            break

    eval_stats = pd.DataFrame(agent.eval(n_episodes=n_evals))
    duration = time() - start_time

    save(agent, env, duration, train_stats, [eval_stats])

    r = eval_stats.reward.mean()
    print('=> reward', r)
    return r


def save(agent, env, duration, train_stats: [dict], eval_stats: [pd.DataFrame]=None, solve_episode=None, output_dir='outputs'):
    date_str = datetime.now().strftime('%d.%m %H.%M')
    h = hash(agent)
    save_dir = '{output_dir}/{agent.__class__.__name__} on {env.__class__.__name__} @ {date_str} [{h}]'.format(**locals())
    os.makedirs(save_dir, exist_ok=True)

    train_df = pd.DataFrame(train_stats)
    train_df.to_csv(save_dir + '/train.csv', index=False)

    tail_avg = train_df.reward.tail(100).mean()
    tail_std = train_df.reward.tail(100).std()
    max_reward = train_df.reward.max()
    final_reward = train_df.reward.iloc[-1]
    n_eps = train_df.episode.iloc[-1]

    if eval_stats:
        eval_df  = pd.concat(eval_stats)
        eval_df .to_csv(save_dir + '/evals.csv', index=False)

        max_reward = eval_df.reward.max()
        final_reward = eval_stats[-1].reward.mean()

    with open(save_dir + '/stats.json', 'w') as f:
        json.dump(dict(
            solve_episode=solve_episode,
            time=duration,
            max_reward=max_reward,
            tail_avg=tail_avg,
            tail_std=tail_std,
            final_reward=final_reward,
            n_eps=n_eps,
        ), f)
    with open(save_dir + '/agent.json', 'w') as f:
        json.dump(agent.config, f)
    with open(save_dir + '/env.json', 'w') as f:
        json.dump(env.config, f)

    try:
        agent.save(save_dir + '/model')
    except Exception as e:
        print('could not save model', e, str(e))
    print('[{agent.seed}] {save_dir} saved'.format(**locals()))
