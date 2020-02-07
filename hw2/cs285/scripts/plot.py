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

prefix = 'run-pg_'


def save(names, split_words, title, xlabel='Step', ylabel='Reward', fig_name=''):
    plt.figure()
    for name in names:
        data = pd.read_csv(dir_name / name)
        label = name.split(split_words[0])[1].split(split_words[1])[0]
        plt.plot(data['Step'], data['Value'], label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(save_dir / fig_name)
    # plt.show()


def prob3():
    sb_names = ['run-pg_sb_no_rtg_dsa_CartPole-v0_24-11-2019_20-27-59-tag-Eval_AverageReturn.csv',
                'run-pg_sb_rtg_dsa_CartPole-v0_24-11-2019_20-46-43-tag-Eval_AverageReturn.csv',
                'run-pg_sb_rtg_na_CartPole-v0_24-11-2019_20-47-45-tag-Eval_AverageReturn.csv']

    lb_names = ['run-pg_lb_no_rtg_dsa_CartPole-v0_24-11-2019_20-48-23-tag-Eval_AverageReturn.csv',
                'run-pg_lb_rtg_dsa_CartPole-v0_24-11-2019_20-50-23-tag-Eval_AverageReturn.csv',
                'run-pg_lb_rtg_na_CartPole-v0_24-11-2019_20-52-23-tag-Eval_AverageReturn.csv']
    title = 'CartPole-v0'
    split_words = [prefix, '_' + title]
    save(sb_names, split_words, title=title, fig_name='3-sb')
    save(lb_names, split_words, title=title, fig_name='3-lb')


def prob4():
    names = ['run-pg_ip_b100_r8e-3_InvertedPendulum-v2_24-11-2019_21-59-38-tag-Eval_AverageReturn.csv']
    title = 'InvertedPendulum-v2'
    split_words = [prefix, '_' + title]
    save(names, split_words, title=title, fig_name='4-ip')


def prob6():
    names = ['run-pg_ll_b40000_r0.005_LunarLanderContinuous-v2_24-11-2019_22-26-20-tag-Eval_AverageReturn.csv']
    title = 'LunarLanderContinuous-v2'
    split_words = [prefix, '_' + title]
    save(names, split_words, title=title, fig_name='6-ll')


def prob7():
    names1 = ['run-pg_hc_b10000_lr0.005_nnbaseline_HalfCheetah-v2_25-11-2019_00-26-28-tag-Eval_AverageReturn.csv',
              'run-pg_hc_b10000_lr0.01_nnbaseline_HalfCheetah-v2_25-11-2019_00-32-37-tag-Eval_AverageReturn.csv',
              'run-pg_hc_b10000_lr0.02_nnbaseline_HalfCheetah-v2_25-11-2019_00-39-30-tag-Eval_AverageReturn.csv',
              'run-pg_hc_b30000_lr0.005_nnbaseline_HalfCheetah-v2_25-11-2019_00-52-56-tag-Eval_AverageReturn.csv',
              'run-pg_hc_b30000_lr0.01_nnbaseline_HalfCheetah-v2_25-11-2019_01-10-02-tag-Eval_AverageReturn.csv',
              'run-pg_hc_b30000_lr0.02_nnbaseline_HalfCheetah-v2_25-11-2019_01-28-06-tag-Eval_AverageReturn.csv',
              'run-pg_hc_b50000_lr0.005_nnbaseline_HalfCheetah-v2_25-11-2019_01-46-10-tag-Eval_AverageReturn.csv',
              'run-pg_hc_b50000_lr0.01_nnbaseline_HalfCheetah-v2_25-11-2019_02-15-44-tag-Eval_AverageReturn.csv',
              'run-pg_hc_b50000_lr0.02_nnbaseline_HalfCheetah-v2_25-11-2019_02-31-34-tag-Eval_AverageReturn.csv']
    title = 'HalfCheetah-v2'
    split_words = [prefix, '_' + title]
    save(names1, split_words, title=title, fig_name='7-hc1')

    names2 = ['run-pg_hc_b50000_r0.02_HalfCheetah-v2_25-11-2019_11-25-00-tag-Eval_AverageReturn.csv',
              'run-pg_hc_b50000_r0.02_nn_HalfCheetah-v2_25-11-2019_11-25-23-tag-Eval_AverageReturn.csv',
              'run-pg_hc_b50000_r0.02_rtg_HalfCheetah-v2_25-11-2019_11-52-58-tag-Eval_AverageReturn.csv',
              'run-pg_hc_b50000_r0.02_rtg_nn_HalfCheetah-v2_25-11-2019_11-53-30-tag-Eval_AverageReturn.csv']
    save(names2, split_words, title=title, fig_name='7-hc2')


if __name__ == "__main__":
    prob3()
    prob4()
    prob6()
    prob7()
