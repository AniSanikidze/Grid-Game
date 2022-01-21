from training import Training
import matplotlib.pyplot as plt  # for graphing our mean rewards over time

def compare_epsilon(env,algorithm,max_steps,num_episodes,gamma,alpha):
   Trained_Agent_eps_decay = Training(env,1,gamma,alpha,algorithm,True)
   Trained_Agent_eps_fixed = Training(env,0.5,gamma,alpha,algorithm,False)
   Trained_Agent_eps_fixed1 = Training(env,0.1,gamma,alpha,algorithm,False)

   eps_decay_info = Trained_Agent_eps_decay.train_agent(num_episodes,"", max_steps)
   eps_fixed_info = Trained_Agent_eps_fixed.train_agent(num_episodes,"", max_steps)
   eps_fixed_info1 = Trained_Agent_eps_fixed1.train_agent(num_episodes,"", max_steps)

   plt.plot(eps_decay_info['ep'],
            eps_decay_info['avg'], marker=".", label="decaying,starting eps=1")
   plt.plot(eps_fixed_info['ep'],
            eps_fixed_info['avg'], marker=".", label="fixed,eps=0.5")
   plt.plot(eps_fixed_info1['ep'],
            eps_fixed_info1['avg'], marker=".", label="fixed,eps=0.1")

   plt.legend(loc=4)
   plots = plt.gcf()

   plt.xlabel("Number of Episodes")
   plt.ylabel("Episode Rewards")
   plt.title("Comparison of constant and decaying epsilons," + algorithm)
   plt.show()
   path = 'plots/'
   plots.savefig(path + 'epsilon_comparison_' + algorithm + '.png', dpi=100)


# def gamma_comparison(env, agent):
#    Trained_Agent_gamma_high = Trained_Agent(env, 1, 0.99, 0.15, agent, False)
#    Trained_Agent_gamma_medium = Trained_Agent(env, 1, 0.5, 0.15, agent, False)
#    Trained_Agent_gamma_low = Trained_Agent(env, 1, 0.1, 0.1, agent, False)

#    rewards_gamma_high = Trained_Agent_gamma_high.train_agent(70000)
#    rewards_gamma_medium = Trained_Agent_gamma_medium.train_agent(70000)
#    rewards_gamma_low = Trained_Agent_gamma_low.train_agent(70000)

#    plt.plot(rewards_gamma_high['ep'],
#             rewards_gamma_high['avg'], marker=".", label="gamma = 0.99")
#    plt.plot(rewards_gamma_medium['ep'],
#             rewards_gamma_medium['avg'], marker=".", label="gamma = 0.7")
#    plt.plot(rewards_gamma_low['ep'],
#             rewards_gamma_low['avg'], marker=".", label="gamma = 0.5")

#    plt.legend(loc=4)
#    plots = plt.gcf()
#    # plt.show()
#    if agent == "Q_learning":
#        plots.savefig('Q_learning_gamma_comparison.png', dpi=100)
#    else:
#       plots.savefig('Sarsa_gamma_comparison.png', dpi=100)

# def alpha_comparison(env,agent):
#     Trained_Agent_alpha_high = Trained_Agent(
#            env, 1, 0.9, 0.9, agent, False)
#     Trained_Agent_alpha_medium = Trained_Agent(env, 1, 0.9, 0.4, agent, False)
#     Trained_Agent_alpha_low = Trained_Agent(env, 1, 0.9, 0.15, agent, False)

#     rewards_alpha_high = Trained_Agent_alpha_high.train_agent(70000)
#     rewards_alpha_medium = Trained_Agent_alpha_medium.train_agent(70000)
#     rewards_alpha_low = Trained_Agent_alpha_low.train_agent(70000)

#     plt.plot(rewards_alpha_high['ep'],
#                 rewards_alpha_high['avg'], marker=".", label="alpha = 0.9")
#     plt.plot(rewards_alpha_medium['ep'],
#                 rewards_alpha_medium['avg'], marker=".", label="alpha = 0.4")
#     plt.plot(rewards_alpha_low['ep'],
#                 rewards_alpha_low['avg'], marker=".", label="alpha = 0.15")

#     plt.legend(loc=4)
#     plots = plt.gcf()
#     # plt.show()
#     if agent == "Q_learning":
#       plots.savefig('Q_learning_alpha_comparison.png', dpi=100)
#     else:
#       plots.savefig('Sarsa_alpha_comparison.png', dpi=100)
