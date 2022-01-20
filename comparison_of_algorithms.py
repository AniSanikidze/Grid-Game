from TrainingAgent import Trained_Agent
import matplotlib.pyplot as plt
from RandomAgent import RandomAgent


def compare_algorithms(env):
    random_agent = RandomAgent(env)
    random_agent_results = random_agent.play(150000)

    q_agent = Trained_Agent(env, 1, 0.9, 0.7, "Q",True)
    q_agent_results = q_agent.train_agent(150000,"",100)

    s_agent = Trained_Agent(env, 1, 0.9, 0.15, "S", True)
    s_agent_results = s_agent.train_agent(150000,"",100)

    plt.plot(q_agent_results['ep'],
             q_agent_results['avg'], marker=".", label="Q-learning")
    plt.plot(s_agent_results['ep'],
             s_agent_results['avg'], marker=".", label="Sarsa")
    plt.plot(random_agent_results['ep'], random_agent_results['avg'],
             marker=".", label="Random")

    plt.legend(loc=1)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Episode Rewards")
    plt.title("Comparison of random agent, Q-learning and Sarsa")
    plots = plt.gcf()
    plt.show()

    plots.savefig('comparison_Q_S_R.png', dpi=100)
