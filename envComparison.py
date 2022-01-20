from TrainingAgent import Trained_Agent
import matplotlib.pyplot as plt
from environment import Grid


def env_comparison(algorithm):
    env_small = Grid(3)
    env_large = Grid(7)

    small_grid_training = Trained_Agent(env_small,1,0.9,0.5,algorithm,True)
    small_grid_training_results = small_grid_training.train_agent(150000,"",100)

    large_grid_training = Trained_Agent(env_large,1,0.9,0.5,algorithm,True)
    large_grid_training_results = large_grid_training.train_agent(150000,"",100)

    plt.plot(small_grid_training_results['ep'],
             small_grid_training_results['avg'], marker=".", label="3x3 Grid")
    plt.plot(large_grid_training_results['ep'],
             large_grid_training_results['avg'], marker=".", label="7x7 Grid")

    plt.legend(loc=4)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Episode Rewards")
    plt.title("Comparison of 3x3 and 7x7 grids")
    plots = plt.gcf()
    plt.show()
    
    if algorithm == "Q":
        plots.savefig('envs_comparison_Q_learning.png', dpi=100)
    else:
        plots.savefig('envs_comparison_Sarsa.png', dpi=100)
