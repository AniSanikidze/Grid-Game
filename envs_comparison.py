from training import Training
import matplotlib.pyplot as plt
from environment import Grid


def env_comparison(algorithm,epsilon,gamma,alpha,decaying_eps,num_episodes,max_steps):
    env_small = Grid(3)
    env_large = Grid(7)

    small_grid_training = Training(env_small,epsilon,gamma,alpha,algorithm,decaying_eps)
    small_grid_training_results = small_grid_training.train_agent(num_episodes,"",max_steps)

    large_grid_training = Training(env_large,epsilon,gamma,alpha,algorithm,decaying_eps)
    large_grid_training_results = large_grid_training.train_agent(num_episodes,"",max_steps)

    plt.plot(small_grid_training_results['ep'],
             small_grid_training_results['avg'], marker=".", label="3x3 Grid")
    plt.plot(large_grid_training_results['ep'],
             large_grid_training_results['avg'], marker=".", label="7x7 Grid")

    plt.legend(loc=4)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Episode Rewards")
    plt.title("Comparison of 3x3 and 7x7 grids," + algorithm)
    plots = plt.gcf()
    plt.show()
    path = 'plots/'
    plots.savefig(path + 'envs_comparison_' + algorithm + '.png', dpi=100)
