import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

class Training:
    def __init__(self, environment, epsilon, gamma, alpha, algorithm, decaying_eps):
        self.environment = environment
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.q_table = {}
        self.create_q_table()
        self.algorithm = algorithm
        self.decaying_eps = decaying_eps

    def create_q_table(self):
        self.q_table = np.zeros((len(self.environment.get_state_space().keys()),
                                len(self.environment.action_space)))

    def save_q_table(self, q_table_to_save):
        path = "q_tables\{}.pkl".format(q_table_to_save)
        with open(path, "wb") as G:
            pickle.dump(self.q_table, G)

    def max_action(self, state, q_table):
        action = np.argmax(q_table[state, :])
        return action

    def choose_action(self, state):
        exploit = random.uniform(0, 1)
        if exploit > self.epsilon:
            action = np.argmax(self.q_table[state, :])
        else:
            action = self.environment.action_space_sample()
        return action

    def decay_epsilon(self, num_epsiodes):
        if self.decaying_eps == True:
            if self.epsilon - 2 / num_epsiodes > 0:
                self.epsilon -= 2 / num_epsiodes
            else:
                self.epsilon = 0.01
            return self.epsilon

    def update_q_table(self, state, new_state, action, reward):
        val = self.q_table[state, action] + self.alpha * (
            reward
            + self.gamma * np.max(self.q_table[new_state, :])
            - self.q_table[state, action]
        )
        self.q_table[state, action] = val

    def get_training_info(
        self,
        num_episode,
        ep_rewards,
        number_of_iterations,
        trajectory,
        starting_position,
        info
    ):
        if num_episode % 5000 == 0:
            print(
                f"Training info of {self.algorithm}: episode {num_episode}, episode reward:{ep_rewards}, number of steps: {number_of_iterations}, epsilon: {self.epsilon}, gamma: {self.gamma}, alpha: {self.alpha} starting position: {starting_position}, trajectory: {trajectory}, {info}"
            )

    def get_training_results(self,ep_num,ep_results,episode_rewards,num_of_iterations):
        if not ep_num % 5000:
            average_reward = sum(episode_rewards[-5000:]) / 5000
            average_num_of_steps = sum(num_of_iterations[-5000:]) / 5000
            ep_results["ep"].append(ep_num)
            ep_results["avg"].append(average_reward)
            ep_results["max"].append(max(episode_rewards[-5000:]))
            ep_results["min"].append(min(episode_rewards[-5000:]))
            ep_results["number of steps"].append(average_num_of_steps)
        return ep_results

    def plot_training_results(self,ep_results,agent):

        plt.style.use("ggplot")

        plt.plot(ep_results["ep"], ep_results["avg"], label="average rewards")
        plt.plot(ep_results["ep"], ep_results["max"], label="max rewards")
        plt.plot(ep_results["ep"], ep_results["min"], label="min rewards")

        plt.plot(
            ep_results["ep"],
            ep_results["number of steps"],
            label="number of steps"
        )
        plt.legend(loc=4)
        plt.xlabel("Number of Episodes")
        plt.title("Training results of " + agent)
        # plots = plt.gcf()
        plt.show()
        # path = 'plots/'
        # if agent == "Q-learning":
        #     plots.savefig(path + 'Training_Results_Q_learning.png', dpi=100)
        # else:
        #     plots.savefig(path + 'Training_Results_Sarsa.png', dpi=100)

    def train_agent(self, num_games,q_table,max_steps):
        num_games = int(num_games)
        env = self.environment
        maximum_steps_per_game = max_steps
        episode_rewards = []
        ep_results = {
            "ep": [],
            "avg": [],
            "min": [],
            "max": [],
            "number of steps": [],
        }
        num_of_iterations = []

        for ep_num in range(num_games):
            done = False
            ep_rewards = 0
            state = env.reset()

            if self.algorithm == "Sarsa":
                action = self.choose_action(state)
            number_of_iterations = 0
            trajectory = []
            starting_position = state

            while not done:
                number_of_iterations += 1
                if number_of_iterations == maximum_steps_per_game:
                    break
                if self.algorithm == "Q-learning":
                    action = self.choose_action(state)
                else:
                    action = self.choose_action(state)

                new_state, reward, done, info = env.step(action)

                self.update_q_table(state, new_state, action, reward)
                state = new_state
                trajectory.append(state)
                ep_rewards += reward

            num_of_iterations.append(number_of_iterations)
            self.get_training_info(
                ep_num,
                ep_rewards,
                number_of_iterations,
                trajectory,
                starting_position,
                info
            )
            self.decay_epsilon(num_games)
            episode_rewards.append(ep_rewards)
            ep_results = self.get_training_results(ep_num,ep_results,episode_rewards,num_of_iterations)

        self.plot_training_results(ep_results,self.algorithm)

        if q_table != "":
            self.save_q_table(q_table)
        return ep_results
