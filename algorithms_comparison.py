from training import Training
import matplotlib.pyplot as plt
from random_agent import Random_Agent

def compare_algorithms(env,num_eps,max_steps,epsilon,gamma,alpha,decaying_eps):
    random_agent = Random_Agent(env)
    random_agent_results = random_agent.play(num_eps)

    q_agent = Training(env, epsilon, gamma, alpha, "Q-learning",decaying_eps)
    q_agent_results = q_agent.train_agent(num_eps,"",max_steps)

    s_agent = Training(env, epsilon, gamma, alpha, "Sarsa", decaying_eps)
    s_agent_results = s_agent.train_agent(num_eps,"",max_steps)

    plt.plot(q_agent_results['ep'],
             q_agent_results['avg'],
             marker=".", label="Q-learning")
    plt.plot(s_agent_results['ep'],
             s_agent_results['avg'],
             marker=".", label="Sarsa")
    plt.plot(random_agent_results['ep'],
             random_agent_results['avg'],
             marker=".", label="Random")

    plt.legend(loc=2)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Episode Rewards")
    plt.title("Comparison of random agent, Q-learning and Sarsa")
    plt.show()