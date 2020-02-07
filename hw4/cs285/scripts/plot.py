import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from pathlib import Path

dir_name = Path(__file__).absolute().parents[1] / 'data/csv/'
save_dir = Path(__file__).absolute().parents[2] / 'results/figs'

params = {'figure.figsize': '10, 5',
          'figure.dpi': 300,
        #   'axes.titlesize': ''
        #   'axes.labelsize': '16',
        #   'xtick.labelsize': '16',
        #   'ytick.labelsize': '13',
        #   'lines.linewidth': '2',
        #   'legend.fontsize': '5',
          'savefig.dpi': 300,
          'savefig.pad_inches': 0.2,
          'savefig.bbox': 'tight'
          }
pylab.rcParams.update(params)


def save(names, split_words, title, labels=None, xlabel='Iteration', ylabel='Eval_AverageReward', fig_name=''):
    plt.figure()
    for i, name in enumerate(names):
        data = pd.read_csv(dir_name / name)
        label = name.split(split_words[0])[1].split(split_words[1])[0]

        plt.plot(data['Step'], data['Value'], label=label if labels is None else labels[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(names) > 1:
        plt.legend()
    plt.savefig(save_dir / fig_name)
    plt.show()


def prob3():
    names = ['run-mb_obstacles_obstacles-cs285-v0_18-01-2020_10-52-39-tag-Eval_AverageReturn.csv',
             'run-mb_reacher_reacher-cs285-v0_18-01-2020_01-12-51-tag-Eval_AverageReturn.csv',
             'run-mb_cheetah_cheetah-cs285-v0_18-01-2020_01-55-05-tag-Eval_AverageReturn.csv']
    titles = ['Obstacles', 'Reacher', 'Cheetah']
    prefix = 'run-mb_'
    split_words = [prefix, '-cs285']
    save([names[0]], split_words, title=titles[0], fig_name='3-1')
    save([names[1]], split_words, title=titles[1], fig_name='3-2')
    save([names[2]], split_words, title=titles[2], fig_name='3-3')


def prob4():
    names = ['run-mb_q5_reacher_horizon5_reacher-cs285-v0_18-01-2020_01-27-04-tag-Eval_AverageReturn.csv',
             'run-mb_q5_reacher_horizon15_reacher-cs285-v0_18-01-2020_01-33-24-tag-Eval_AverageReturn.csv',
             'run-mb_q5_reacher_horizon30_reacher-cs285-v0_18-01-2020_01-49-01-tag-Eval_AverageReturn.csv']
    title = 'Reacher'
    prefix = 'run-mb_q5_reacher_'
    split_words = [prefix, '_reacher']
    save(names, split_words, title=title, fig_name='4-1')

    names = ['run-mb_q5_reacher_numseq100_reacher-cs285-v0_18-01-2020_02-18-09-tag-Eval_AverageReturn.csv',
             'run-mb_q5_reacher_numseq1000_reacher-cs285-v0_18-01-2020_02-27-38-tag-Eval_AverageReturn.csv']
    save(names, split_words, title=title, fig_name='4-2')

    names = ['run-mb_q5_reacher_ensemble1_reacher-cs285-v0_18-01-2020_02-38-22-tag-Eval_AverageReturn.csv',
             'run-mb_q5_reacher_ensemble3_reacher-cs285-v0_18-01-2020_02-42-28-tag-Eval_AverageReturn.csv',
             'run-mb_q5_reacher_ensemble5_reacher-cs285-v0_18-01-2020_02-53-10-tag-Eval_AverageReturn.csv']
    save(names, split_words, title=title, fig_name='4-3')


if __name__ == "__main__":
    prob3()
    prob4()
