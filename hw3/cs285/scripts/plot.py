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


def save(names, split_words, title, labels=None, xlabel='Step', ylabel='Reward', fig_name=''):
    plt.figure()
    for i, name in enumerate(names):
        data = pd.read_csv(dir_name / name)
        label = name.split(split_words[0])[1].split(split_words[1])[0]

        plt.plot(data['Step'], data['Value'], label=label if labels is None else labels[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(save_dir / fig_name)
    plt.show()


def average(names):
    y = []
    for name in names:
        data = pd.read_csv(dir_name / name)
        y.append(data['Value'])
    x = pd.read_csv(dir_name / name)['Step']
    y = np.array(y)
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    return x, y_mean, y_std


def prob1():
    names = ['run-dqn_q1_PongNoFrameskip-v4_26-11-2019_00-07-30-tag-Train_AverageReturn.csv']
    title = 'PongNoFrameskip-v4'
    prefix = 'run-dqn_'
    split_words = [prefix, '_' + title]
    save(names, split_words, title=title, fig_name='1')


def prob2():
    dqn_names = ['run-dqn_q2_dqn_1_LunarLander-v2_26-11-2019_12-55-21-tag-Train_AverageReturn.csv',
                 'run-dqn_q2_dqn_2_LunarLander-v2_26-11-2019_12-56-06-tag-Train_AverageReturn.csv',
                 'run-dqn_q2_dqn_3_LunarLander-v2_26-11-2019_12-59-53-tag-Train_AverageReturn.csv']

    ddqn_names = ['run-dqn_double_q_q2_doubledqn_1_LunarLander-v2_26-11-2019_13-21-45-tag-Train_AverageReturn.csv',
                  'run-dqn_double_q_q2_doubledqn_2_LunarLander-v2_26-11-2019_13-22-00-tag-Train_AverageReturn.csv',
                  'run-dqn_double_q_q2_doubledqn_3_LunarLander-v2_26-11-2019_13-22-11-tag-Train_AverageReturn.csv']
    title = 'LunarLander-v2'
    prefix = 'run-dqn_'
    split_words = [prefix, '_' + title]
    x_1, y_mean_1, y_std_1 = average(dqn_names)
    x_2, y_mean_2, y_std_2 = average(ddqn_names)

    plt.figure()
    plt.plot(x_1, y_mean_1, label='dqn')
    plt.plot(x_2, y_mean_2, label='ddqn')
    plt.legend()
    plt.savefig(save_dir / '2')
    plt.show()


def prob3():
    names = ['run-dqn_q3_batch16_PongNoFrameskip-v4_26-11-2019_14-42-44-tag-Train_AverageReturn.csv',
             'run-dqn_q1_PongNoFrameskip-v4_26-11-2019_00-07-30-tag-Train_AverageReturn.csv',
             'run-dqn_q3_batch64_PongNoFrameskip-v4_26-11-2019_17-51-28-tag-Train_AverageReturn.csv',
             'run-dqn_q3_128_PongNoFrameskip-v4_27-11-2019_03-00-49-tag-Train_AverageReturn.csv']
    title = 'PongNoFrameskip-v4'
    labels = ['batch16', 'batch32', 'batch64', 'batch128']
    prefix = 'run-dqn_'
    split_words = [prefix, '_' + title]
    save(names, split_words, labels=labels, title=title, fig_name='3')


def prob4():
    names = ['run-ac_1_1_CartPole-v0_29-11-2019_00-33-05-tag-Eval_AverageReturn.csv',
             'run-ac_1_100_CartPole-v0_29-11-2019_00-35-39-tag-Eval_AverageReturn.csv',
             'run-ac_10_10_CartPole-v0_29-11-2019_00-36-16-tag-Eval_AverageReturn.csv',
             'run-ac_100_1_CartPole-v0_29-11-2019_00-34-56-tag-Eval_AverageReturn.csv']
    prefix = 'run-ac_'
    title = 'CartPole-v0'
    split_words = [prefix, '_' + title]
    save(names, split_words, title=title, fig_name='4')


def prob5():
    names1 = ['run-ac_10_10_InvertedPendulum-v2_29-11-2019_00-55-13-tag-Eval_AverageReturn.csv']
    prefix = 'run-ac_'
    title = 'InvertedPendulum-v2'
    split_words = [prefix, '_' + title]
    save(names1, split_words, title=title, fig_name='5-1')

    names2 = ['run-ac_10_10_HalfCheetah-v2_29-11-2019_00-58-52-tag-Eval_AverageReturn.csv']
    title = 'HalfCheetah-v2'
    split_words = [prefix, '_' + title]
    save(names2, split_words, title=title, fig_name='5-2')


if __name__ == "__main__":
    prob1()
    prob2()
    prob3()
    prob4()
    prob5()
