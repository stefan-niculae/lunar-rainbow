""" Visualization functions """

import matplotlib.pyplot as plt
from typing import Union
from scipy.signal import savgol_filter
from matplotlib.ticker import FuncFormatter
import pandas as pd

def readable_big_number(x: Union[int, float], precision=1, threshold=1) -> str:
    """
    >>> readable_big_number(1_000)
    '1k'
    >>> readable_big_number(20_123, precision=1)
    '20.1k'
    """
    if x == 0:
        return '0'

    was_negative = x < 0
    x = abs(x)

    if x < 10**3 * threshold:
        suffix = ''
    elif x < 10**6 * threshold:
        x /= float(10**3)
        suffix = 'k'
    elif x < 10**9 * threshold:
        x /= float(10**6)
        suffix = 'm'
    else:
        x /= float(10**9)
        suffix = 'b'

    fmt = '{:.' + str(precision) + 'f}'
    nr = fmt.format(x)

    nr = nr.lstrip('0')  # remove leading zeros
    if precision > 0:  # otherwise we strip useful zeros
        nr = nr.rstrip('0').rstrip('.')  # remove trailing zeros

    prefix = '-' if was_negative else ''
    return prefix + nr + suffix

def training_progress(dfs: [pd.DataFrame], window_sizes: [int], names: [str], bands=False, column='reward') -> [[float]]:
    """ larger `window_size` => smoother curve """

    # sort them by the mean value so that the labels are in the same order as the lines
    smoothed_traces = []
    for i, (df, window_size, name) in enumerate(zip(dfs, window_sizes, names)):
        if window_size % 2 == 0:
            window_size += 1  # must be odd

        color = f'C{names.index(name) % 10}'

        stats = df.groupby('episode').describe()[column]
        mean = stats['mean']

        if bands:
            mins, maxs = stats['min'], stats['max']
            mean.plot(alpha=1, color=color)
            plt.fill_between(stats.index, mins, maxs, alpha=.1, color=color)
        elif window_size:
            ws = window_size
            if window_size > len(mean):
                ws = len(mean) // 2
                if ws % 2 == 0:
                    ws += 1  # must be odd
            smoothed = savgol_filter(mean, ws, polyorder=1)
            smoothed_traces.append(smoothed)
            plt.plot(mean.index, smoothed, alpha=1, label=name, color=color)
            mean.plot(alpha=.05, label='', color=color)

    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: readable_big_number(x)))
    # plt.axhline(c='grey')

    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.legend()  # bottom right

    return smoothed_traces
