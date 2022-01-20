import numpy as np
import matplotlib.pyplot as plt  # for graphing our mean rewards over time

class RandomAgent:
    def __init__(self,env):
        self.env = env

    def choose_action(self):
        random_action = np.random.choice((0,1,2,3))
        return random_action

    def get_random_training_info(self, num_episode, ep_rewards, number_of_iterations, trajectory, starting_position, info):
        if num_episode % 5000 == 0:
            print(
                f'Training info of random agent: episode {num_episode}, episode reward:{ep_rewards}, number of steps: {number_of_iterations}, starting position: {starting_position}, trajectory: {trajectory}, {info}')

    def play(self,num_games):
        maximum_steps_per_game = 500
        episode_rewards = []
        aggr_ep_rewards = {"ep": [], "avg": [], "min": [],
                           "max": [], "epsilon": [], "Number of steps": []}
        # total_rewards = np.zeros(num_games)
        num_of_iterations = []

        for i in range(num_games):
            done = False
            ep_rewards = 0
            state = self.env.reset()
            number_of_iterations = 0
            trajectory = []
            starting_position = self.env.getAgentPosition()

            while not done:
                number_of_iterations += 1
                if(number_of_iterations == maximum_steps_per_game):
                    break
                action = self.choose_action()
                new_state, reward, done, info = self.env.step(action)
                trajectory.append(state)
                state = new_state
                ep_rewards += reward

            # total_rewards[i] = ep_rewards
            num_of_iterations.append(number_of_iterations)
            self.get_random_training_info(
                i, ep_rewards, number_of_iterations, trajectory, starting_position, info)
            # self.decay_epsilon(num_games)
            episode_rewards.append(ep_rewards)
            if not i % 5000:
                average_reward = sum(episode_rewards[-5000:])/5000
                average_num_of_steps = sum(num_of_iterations[-5000:])/5000
                aggr_ep_rewards['ep'].append(i)
                aggr_ep_rewards['avg'].append(average_reward)
                aggr_ep_rewards['max'].append(max(episode_rewards[-5000:]))
                aggr_ep_rewards['min'].append(min(episode_rewards[-5000:]))
                # aggr_ep_rewards["epsilon"].append(self.epsilon)
                aggr_ep_rewards["Number of steps"].append(average_num_of_steps)
        #     print(
        #         f'Episode: {i:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {self.epsilon:>1.2f}')
        # # plt.plot(total_rewards,label='Reward Per Episode')
        plt.style.use('ggplot')

        plt.plot(aggr_ep_rewards['ep'],
                 aggr_ep_rewards['avg'], label="average rewards")

        plt.plot(aggr_ep_rewards['ep'],
                 aggr_ep_rewards['max'], label="max rewards")
        plt.plot(aggr_ep_rewards['ep'],
                 aggr_ep_rewards['min'], label="min rewards")
        # plt.plot(aggr_ep_rewards["ep"],aggr_ep_rewards["epsilon"], label = "epsilon")
        plt.plot(
            aggr_ep_rewards["ep"], aggr_ep_rewards["Number of steps"], label="Number of steps")
        plt.legend(loc=4)

        # # plt.plot(num_of_iterations,label="Steps Per Episode")
        # # plt.legend(loc="lower right")
        plt.xlabel('Number of Episodes ->')
        plt.ylabel("Episode Rewards")
        # # plt.title('Training progress')
        # # plt.show()
        plots = plt.gcf()
        plt.show()
        plots.savefig('Random_agent_plots.png', dpi=100)
        # plotting.plot_episode_stats(aggr_ep_rewards)
        # self.save(self.agent)
        return aggr_ep_rewards
