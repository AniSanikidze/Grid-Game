import re
from unittest import result
from TrainingAgent import Trained_Agent
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
import pickle

# def save(parameter,results):
#    if (parameter == "e"):
#       with open("epsilon_results.pkl", 'wb') as G:
#         pickle.dump(results, G)


# def load(parameter):
#     if (parameter == "e"):
#         with open("epsilon_results.pkl", 'rb') as G:
#             return pickle.load(G)
   #  else:
   #      with open("Sarsa-Q-table.pkl", "rb") as G:
   #          return pickle.load(G)

   # else:
   #    with open("Sarsa-Q-table.pkl", "wb") as G:
   #       pickle.dump(results, G)

def ExploitationVSExploration(env,agent):
   Trained_Agent_eps_decay = Trained_Agent(env,1,0.9,0.15,agent,False)
   Trained_Agent_eps_fixed = Trained_Agent(env,0.5,0.9,0.15,agent,True)
   Trained_Agent_eps_fixed1 = Trained_Agent(env,0.1,0.9,0.1,agent,True)

   eps_decay_info = Trained_Agent_eps_decay.train_agent(100000)
   eps_fixed_info = Trained_Agent_eps_fixed.train_agent(100000)
   eps_fixed_info1 = Trained_Agent_eps_fixed1.train_agent(100000)
   eps_results = [eps_decay_info,eps_fixed_info,eps_fixed_info1]
   # save("e",eps_results)

   # results = load("e")
   e = eps_results[0]
   ef = eps_results[1]
   ef1 = eps_results[2]

   # open_file = open(file_name, "rb")
   # loaded_list = pickle.load(open_file)
   # open_file.close()

   plt.plot(e['ep'],
            e['avg'], marker=".", label="decaying,starting eps=1")
   plt.plot(ef['ep'],
            ef['avg'], marker=".", label="fixed,eps=0.5")
   plt.plot(ef1['ep'],
            ef1['avg'], marker=".", label="fixed,eps=0.1")

   plt.legend(loc=3)
   plots = plt.gcf()
   # plt.show()
   plt.xlabel("Number of Episodes ->")
   plt.ylabel("Episode Rewards")
   if agent == "Q_learning":
            plots.savefig('Q_learning_eps_comparison.png', dpi=100)
   else:
      plots.savefig('Sarsa_eps_comparison.png', dpi=100)


def gamma_comparison(env, agent):
   Trained_Agent_gamma_high = Trained_Agent(env, 1, 0.99, 0.15, agent, False)
   Trained_Agent_gamma_medium = Trained_Agent(env, 1, 0.5, 0.15, agent, False)
   Trained_Agent_gamma_low = Trained_Agent(env, 1, 0.1, 0.1, agent, False)

   rewards_gamma_high = Trained_Agent_gamma_high.train_agent(70000)
   rewards_gamma_medium = Trained_Agent_gamma_medium.train_agent(70000)
   rewards_gamma_low = Trained_Agent_gamma_low.train_agent(70000)

   plt.plot(rewards_gamma_high['ep'],
            rewards_gamma_high['avg'], marker=".", label="gamma = 0.99")
   plt.plot(rewards_gamma_medium['ep'],
            rewards_gamma_medium['avg'], marker=".", label="gamma = 0.7")
   plt.plot(rewards_gamma_low['ep'],
            rewards_gamma_low['avg'], marker=".", label="gamma = 0.5")

   plt.legend(loc=4)
   plots = plt.gcf()
   # plt.show()
   if agent == "Q_learning":
       plots.savefig('Q_learning_gamma_comparison.png', dpi=100)
   else:
      plots.savefig('Sarsa_gamma_comparison.png', dpi=100)

def alpha_comparison(env,agent):
    Trained_Agent_alpha_high = Trained_Agent(
           env, 1, 0.9, 0.9, agent, False)
    Trained_Agent_alpha_medium = Trained_Agent(env, 1, 0.9, 0.4, agent, False)
    Trained_Agent_alpha_low = Trained_Agent(env, 1, 0.9, 0.15, agent, False)

    rewards_alpha_high = Trained_Agent_alpha_high.train_agent(70000)
    rewards_alpha_medium = Trained_Agent_alpha_medium.train_agent(70000)
    rewards_alpha_low = Trained_Agent_alpha_low.train_agent(70000)

    plt.plot(rewards_alpha_high['ep'],
                rewards_alpha_high['avg'], marker=".", label="alpha = 0.9")
    plt.plot(rewards_alpha_medium['ep'],
                rewards_alpha_medium['avg'], marker=".", label="alpha = 0.4")
    plt.plot(rewards_alpha_low['ep'],
                rewards_alpha_low['avg'], marker=".", label="alpha = 0.15")

    plt.legend(loc=4)
    plots = plt.gcf()
    # plt.show()
    if agent == "Q_learning":
      plots.savefig('Q_learning_alpha_comparison.png', dpi=100)
    else:
      plots.savefig('Sarsa_alpha_comparison.png', dpi=100)
