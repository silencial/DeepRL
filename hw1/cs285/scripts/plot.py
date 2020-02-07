import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

save_dir = Path(__file__).absolute().parents[2] / 'results/figs'


def prob1():
    envname = 'Ant-v2'
    num_iter = list(range(0, 2101, 300))
    mean = [-388.25, 80.08, 903.87, 2236.70, 3537.81, 4052.48, 4315.12, 4448.00]
    std = [1039.12, 233.12, 547.58, 1160.82, 663.74, 159.67, 42.78, 104.43]

    expert_mean = [4713.65] * len(num_iter)
    expert_std = [12.20] * len(num_iter)

    # plt.figure(figsize=(1920 / 72, 1080 / 72))
    plt.figure()
    plt.errorbar(num_iter, mean, std, marker='o', capsize=8, linestyle='--', label='Behavior Cloning')
    plt.errorbar(num_iter, expert_mean, expert_std, marker='o', capsize=8, linestyle='--', label='Expert')
    plt.xlabel('Number of training')
    plt.ylabel('Reward')
    plt.title(envname)
    plt.legend()

    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)

    plt.savefig(save_dir / 'bc1.png', bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.show()


def prob2():
    envname = 'Humanoid-v2'
    num_iter = list(range(0, 20))
    mean = [441.98, 285.68, 360.19, 469.41, 491.20, 817.60, 1859.49, 946.65, 987.78, 1592.18,
            1998.80, 1917.24, 1595.01, 10062.73, 2320.06, 1974.80, 5886.82, 7478.71, 2123.53, 10257.44]
    std = [185.46, 16.14, 38.38, 154.77, 48.87, 119.71, 444.86, 163.30, 54.54, 27.21,
        0, 0, 3.82, 0, 0, 0, 0, 0, 1814.61, 0]

    expert_mean = [10344.52] * len(num_iter)
    expert_std = [20.98] * len(num_iter)

    imitation_mean = [339.88] * len(num_iter)
    imitation_std = [77.87] * len(num_iter)

    ax = plt.figure().gca()
    plt.errorbar(num_iter, mean, std, marker='o', capsize=8, linestyle='--', label='DAgger')
    plt.errorbar(num_iter, expert_mean, expert_std, marker='o', capsize=8, linestyle='--', label='Expert')
    plt.errorbar(num_iter, imitation_mean, imitation_std, marker='o', capsize=8, linestyle='--', label='Behavior Cloning')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Iters')
    plt.ylabel('Reward')
    plt.title(envname)
    plt.legend()

    plt.savefig(save_dir / 'dagger.png', bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.show()


if __name__ == "__main__":
    prob1()
    prob2()
