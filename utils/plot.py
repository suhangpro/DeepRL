# Adapted from  https://github.com/openai/baselines/blob/master/baselines/results_plotter.py

import numpy as np
import component
import os
import re

class Plotter:
    COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
              'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
              'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

    X_TIMESTEPS = 'timesteps'
    X_EPISODES = 'episodes'
    X_WALLTIME = 'walltime_hrs'

    def __init__(self):
        pass

    def rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def window_func(self, x, y, window, func):
        yw = self.rolling_window(y, window)
        yw_func = func(yw, axis=-1)
        return x[window - 1:], yw_func

    def ts2xy(self, ts, xaxis):
        if xaxis == Plotter.X_TIMESTEPS:
            x = np.cumsum(ts.l.values)
            y = ts.r.values
        elif xaxis == Plotter.X_EPISODES:
            x = np.arange(len(ts))
            y = ts.r.values
        elif xaxis == Plotter.X_WALLTIME:
            x = ts.t.values / 3600.
            y = ts.r.values
        else:
            raise NotImplementedError
        return x, y

    def load_results(self, dirs, max_timesteps=1e8, x_axis=X_TIMESTEPS, episode_window=100):
        tslist = []
        for dir in dirs:
            ts = component.load_monitor_log(dir)
            ts = ts[ts.l.cumsum() <= max_timesteps]
            tslist.append(ts)
        xy_list = [self.ts2xy(ts, x_axis) for ts in tslist]
        if episode_window:
            xy_list = [self.window_func(x, y, episode_window, np.mean) for x, y in xy_list]
        return xy_list

    def average(self, xy_list, bin, max_timesteps, top_k=0):
        if top_k:
            perf = [np.max(y) for _, y in xy_list]
            top_k_runs = np.argsort(perf)[-top_k:]
            new_xy_list = []
            for r, (x, y) in enumerate(xy_list):
                if r in top_k_runs:
                    new_xy_list.append((x, y))
            xy_list = new_xy_list
        new_x = np.arange(0, max_timesteps, bin)
        new_y = []
        for x, y in xy_list:
            new_y.append(np.interp(new_x, x, y))
        return new_x, np.asarray(new_y)

    def plot_results(self, dirs, max_timesteps=1e8, x_axis=X_TIMESTEPS, episode_window=100, title=None, labels=None):
        import matplotlib.pyplot as plt
        plt.ticklabel_format(axis='x', style='sci', scilimits=(1, 1))
        xy_list = self.load_results(dirs, max_timesteps, x_axis, episode_window)
        if type(labels) == str:
            labels = [labels] * len(dirs)
        elif labels is None:
            labels = dirs
        for (i, (x, y)) in enumerate(xy_list):
            color = Plotter.COLORS[i]
            plt.plot(x, y, color=color, label=labels[i])
        plt.legend()
        plt.xlabel(x_axis)
        plt.ylabel("Episode Rewards")
        if title is not None:
            plt.title(title)

    def load_log_dirs(self, pattern, negative_pattern=' ', root='./log', **kwargs):
        dirs = [item[0] for item in os.walk(root)]
        leaf_dirs = []
        for i in range(len(dirs)):
            if i + 1 < len(dirs) and dirs[i + 1].startswith(dirs[i]):
                continue
            leaf_dirs.append(dirs[i])
        names = []
        p = re.compile(pattern)
        np = re.compile(negative_pattern)
        for dir in leaf_dirs:
            if p.match(dir) and not np.match(dir):
                names.append(dir)
                print(dir)

        return sorted(names)

if __name__ == '__main__':
    """Usage: plot.py LOG_PATH FIG_PATH"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('log_root')
    parser.add_argument('out_path')
    parser.add_argument('--pattern', default='', type=str)
    parser.add_argument('--neg_pattern', default=' ', type=str)
    args = parser.parse_args()
    plt.figure()
    plotter = Plotter()
    plotter.plot_results(plotter.load_log_dirs(args.pattern, args.neg_pattern, root=args.log_root))
    plt.savefig(args.out_path)
