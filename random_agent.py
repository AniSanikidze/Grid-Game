import numpy as np
from training import Training

class Random_Agent:
    def __init__(self,env):
        self.env = env

    def choose_action(self):
        random_action = np.random.choice((0,1,2,3))
        return random_action

    def get_random_training_info(self, num_episode, ep_rewards,
                                 number_of_iterations, trajectory,
                                 starting_position, info):
        if num_episode % 5000 == 0:
            print(
                f'Training info of random agent: episode {num_episode},' +
                 f'episode reward:{ep_rewards}, number of steps: {number_of_iterations},' +
                 f'starting position: {starting_position}, trajectory: {trajectory}, {info}')

    def play(self,num_games):
        maximum_steps_per_game = 500
        episode_rewards = []
        ep_results = {"ep": [], "avg": [], "min": [],
                      "max": [], "number of steps": []}
        num_of_iterations = []

        for ep_num in range(num_games):
            done = False
            ep_rewards = 0
            state = self.env.reset()
            number_of_iterations = 0
            trajectory = []
            starting_position = self.env.get_agent_position()

            while not done:
                number_of_iterations += 1
                if(number_of_iterations == maximum_steps_per_game):
                    break
                action = self.choose_action()
                new_state, reward, done, info = self.env.step(action)
                trajectory.append(state)
                state = new_state
                ep_rewards += reward

            num_of_iterations.append(number_of_iterations)
            self.get_random_training_info(ep_num, ep_rewards,
                                         number_of_iterations,
                                         trajectory, starting_position, info)
            episode_rewards.append(ep_rewards)
            Training.get_training_results(Training,ep_num,ep_results,
                                          episode_rewards,num_of_iterations)

        Training.plot_training_results(Training,ep_results,"Random agent")
        return ep_results
