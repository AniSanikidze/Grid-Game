from training import Training
import matplotlib.pyplot as plt

def compare_epsilon(env,algorithm,max_steps,num_episodes,gamma,alpha):
   Trained_Agent_eps_decay = Training(env,1,gamma,alpha,algorithm,True)
   Trained_Agent_eps_fixed = Training(env,0.5,gamma,alpha,algorithm,False)
   Trained_Agent_eps_fixed1 = Training(env,0.1,gamma,alpha,algorithm,False)

   eps_decay_results = Trained_Agent_eps_decay.train_agent(num_episodes,"", max_steps)
   eps_decay_results = Trained_Agent_eps_fixed.train_agent(num_episodes,"", max_steps)
   eps_decay_results1 = Trained_Agent_eps_fixed1.train_agent(num_episodes,"", max_steps)

   plt.plot(eps_decay_results['ep'],
            eps_decay_results['avg'],
            marker=".",
            label="decaying,starting eps=1")
   plt.plot(eps_decay_results['ep'],
            eps_decay_results['avg'],
            marker=".",
            label="fixed,eps=0.5")
   plt.plot(eps_decay_results1['ep'],
            eps_decay_results1['avg'],
            marker=".",
            label="fixed,eps=0.1")

   plt.legend(loc=4)
   plt.xlabel("Number of Episodes")
   plt.ylabel("Episode Rewards")
   plt.title("Comparison of constant and decaying epsilons," + algorithm)
   plt.show()