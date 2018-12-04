import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode
import seaborn as sns
sns.set_style('whitegrid')
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
sns.set_palette(sns.xkcd_palette(colors))
import math

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

TOT_TIMESTEPS = int(1e6)
X_TIMESTEPS = 'timesteps'
X_SAMPLES = 'num_samples'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_func(x, y, window, func):
    ys = np.concatenate((np.ones(window-1)*y[0], y), axis=0)
    yw = rolling_window(ys, window)
    yw_func = func(yw, axis=-1)
    return yw_func

# from https://www.dropbox.com/s/2chw5biyvmd6xvf/ppo-gym-v2.zip?dl=0&file_subpath=%2Fppo-gym-v2%2Fplot.py
# and https://github.com/openai/gym/pull/834
def smooth_reward_curve(x, y):

    halfwidth = int(np.ceil(len(x) / 60))
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='same')
    return xsmoo, ysmoo

def ts2xy(ts, xaxis, style=0):
    if xaxis == X_SAMPLES:
        x = ts.num_samples.values
        y = ts.stoc_pol_mean.values
        x_vals = np.arange(0, TOT_TIMESTEPS, 100)
        y_vals = np.interp(x_vals, x, y)
        if style:
            x_vals, y_vals = smooth_reward_curve(x_vals, y_vals) #         Call if PPO-like
        x, y = x_vals, y_vals
    elif xaxis == X_EPISODES:
        x = np.arange(len(ts))
        y = ts.r.values
    elif xaxis == X_WALLTIME:
        x = ts.t.values / 3600.
        y = ts.r.values
    else:
        raise NotImplementedError
    return x, y

def load_monitor(log, style):

    import pandas
    import glob
    import os.path as osp
    monitor_file = glob.glob(osp.join(log, "log.csv"))[0]
    with open(monitor_file, 'rt') as fh:
        df = pandas.read_csv(fh, index_col=None)
    df.sort_values('num_samples', inplace=True)
    df.reset_index(inplace=True)
    x, y = ts2xy(df, 'num_samples', style=style)

    return x, y


def plot_results(games, log_dir, seeds, labels):
    color_list = ["#feb308", "#3778bf", "#028f1e"]           #Berkeley Colors
    plt.figure(figsize=(9,5))
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 0.5
    plt.rcParams["font.sans-serif"] = "Helvetica"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["ytick.labelsize"] = "large"
    plt.rcParams["xtick.labelsize"] = "small"
    plt.rcParams["font.size"] = 8.3
    envs_per_row = 3
    plot_style = [1]
    for j in plot_style:
        for i, game in enumerate(games):
            ax = plt.subplot(len(plot_style) * math.ceil(len(games) / float(envs_per_row)), envs_per_row, (len(plot_style)-1)*j*len(games) + (i+1))
            for col_i, label in enumerate(labels):
                ys = []
                for seed in seeds:
                    log = '%s/%s-v2/%s/%d/logs' % (log_dir, game,
                                                label,
                                                seed)
                    print(log)
                    x, y =  load_monitor(log, j)
                    ys.append(y)
                if not j:
                    ys = [window_func(x, y, EPISODES_WINDOW, np.mean) for y in ys]
                    y_mean = np.mean(ys, axis=0)
                    y_std = np.std(ys, axis=0)
                    y_ste = y_std/np.sqrt(len(ys))
                    plt.plot(x, y_mean, label=label, color=color_list[col_i])
                    plt.fill_between(x, y_mean - y_ste, y_mean + y_ste, alpha=0.25, color=color_list[col_i], linewidth=0.0)
                else:
                    # # # # PPO-like
                    y_median = np.nanmean(np.array(ys), axis=0)
                    if label == 'KLPPO':
                        ac_label = 'PPO (Adaptive KL)'
                    elif label == 'PPO':
                        ac_label = 'PPO (Clip)'
                    else:
                        ac_label = label
                    plt.plot(x, y_median, label=ac_label, color=color_list[col_i], linewidth=1.5)  #if ril_update==0 else labels[i]+" "+str(lr)
                    plt.fill_between(x, np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25, color=color_list[col_i], linewidth=0.0)
                ax.set_aspect('auto')
            tick_fractions = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
            ticks = tick_fractions * TOT_TIMESTEPS
            tick_names = ["%.1fM"%(tick/float(1e6)) for tick in ticks]
            tick_names[0] = "0"
            tick_names[-1] = "%dM"%(TOT_TIMESTEPS/float(1e6))
            plt.title(game)
            plt.xticks(ticks, tick_names)
            plt.xlim(0, TOT_TIMESTEPS)
    plt.tight_layout()
    plt.legend(frameon=True, loc=4)
    plt.savefig('results.png', dpi=300, transparent=True)
    plt.close()

def main():

    games = [
        'Ant',
        'HalfCheetah',
        'Hopper',
        'Reacher',
        'Swimmer',
        'Walker2d',
        ]

    log_dir = 'results'

    seeds = [0, 1, 2]
    labels = [
            "KLPPO",
            "PPO",
            ]
    plot_results(games, log_dir, seeds, labels)

if __name__ == '__main__':
    main()
