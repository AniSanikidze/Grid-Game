import argparse
from xmlrpc.client import boolean
from TrainingAgent import Trained_Agent
from environment import Grid
from play import play
from comparison_of_algorithms import compare_algorithms
from envComparison import env_comparison

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('keyword', type=str,
                    help='enter one of the keywords: train, play, compare_algorithms, compare_envs')
parser.add_argument('--algorithm', type=str,
                    help='enter the chosen algorithm Q for Q-learning or S for Sarsa, (example: %(default)s)')
parser.add_argument('--q_table', type=str,
                    help='enter the name of the saved Q-table (example: Q_learning_Q-table.pkl')
parser.add_argument('--num_eps', type=int,
                    help='enter the number of episodes')
parser.add_argument('--env_size', type=int, default=7,
                    help='enter the size of the grid (default: %(default)s)')
parser.add_argument('--eps', type=float, default=1,
                    help='enter the value of starting epsilon (default: %(default)s)')
parser.add_argument('--decaying_eps', type=boolean, default=True,
                    help='enter the True for e-greedy policy, otherwise the epsilon value will be fixed (default: %(default)s)')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='enter the value of gamma (default: %(default)s)')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='enter the value of alpha (default: %(default)s)')
parser.add_argument('--max_steps', type=int, default=100,
                    help='enter the maximum number of steps that agent can take per episode (default: %(default)s)')

args = parser.parse_args()
env = Grid(args.env_size)
eps = args.eps
gamma = args.gamma
alpha = args.alpha
num_eps = args.num_eps
decaying_eps = args.decaying_eps
q_table = args.q_table
algorithm = args.algorithm
max_steps = args.max_steps
Trained_agent = Trained_Agent(env,eps,gamma,alpha,algorithm,decaying_eps)
if args.keyword == "train":
    Trained_agent.train_agent(num_eps,q_table,max_steps)
elif args.keyword == "play":
    play(env,Trained_agent,q_table,num_eps,max_steps)
elif args.keyword == "compare_algorithms":
    compare_algorithms(env)
elif args.keyword == "compare_envs":
    env_comparison(algorithm)



# import argparse
# # from operator import le
# # import numpy as np
# from environment import Actions, Grid
# import matplotlib.pyplot as plt
# from RandomAgent import RandomAgent
# from TrainingComparison import comparison
# # from Q_Agent import Q_Agent
# import numpy as np  # for array stuff and random
# from PIL import Image  # for creating visual of our env
# import cv2  # for showing our visual live
# import matplotlib.pyplot as plt  # for graphing our mean rewards over time
# import pickle  # to save/load Q-Tables
# from matplotlib import style  # to make pretty charts because it matters.
# import time  # using this to keep track of our saved Q-Tables.
# from UI import display_ui
# from EpsilonComparison import ExploitationVSExploration,gamma_comparison,alpha_comparison
# # from Sasra import Sarsa_Agent
# from TrainingAgent import Trained_Agent
# from envComparison import env_comparison


# parser = argparse.ArgumentParser(description='Grid Game')
# parser.add_argument('keyword','keyword', type=str,help='enter one of the keywords: train, play, compare')
# args = parser.parse_args()

# # from TrainingComparison import comparison
# # from EpsilonComparison import (
# #     ExploitationVSExploration,
# #     alpha_comparison,
# #     gamma_comparison,
# # )
# import sys


# def load(agent):
#     if agent == "Q-learning":
#         with open("Q-learning-Q-table.pkl", "rb") as G:
#             return pickle.load(G)
#     else:
#         with open("Sarsa-Q-table.pkl", "rb") as G:
#             return pickle.load(G)


# def play(agent, env, num_games, maximum_steps_per_game):

#     env = env
#     # maximum_steps_per_game = 100
#     # total_rewards = np.zeros(num_games)
#     Q = load(agent)
#     num_games = int(num_games)
#     # env.render()
#     import time

#     for i in range(1, num_games + 1):
#         done = False
#         ep_rewards = 0
#         state = env.reset()
#         # display_ui(done, env, agent)
#         number_of_iterations = 0
#         trajectory = []
#         # starting_position = env.getAgentPosition()
#         print(f"****EPISODE {i}****")

#         while number_of_iterations != maximum_steps_per_game:
#             number_of_iterations += 1

#             # if trained agent, then max_action, else Random agent choose_action.
#             action = Trained_Agent.max_action(Trained_Agent, state, Q)

#             if action == Actions.NORTH:
#                 print("North")
#             elif action == Actions.SOUTH:
#                 print("South")
#             elif action == Actions.WEST:
#                 print("West")
#             elif action == Actions.EAST:
#                 print("East")
#             # action = agent.choose_action(state)
#             new_state, reward, done, info = env.step(action)
#             # self.update_q_table(state, new_state, action, reward)
#             state = new_state
#             display_ui(done, env, agent)
#             trajectory.append(state)
#             ep_rewards += reward

#             # total_rewards[i] = ep_rewards
#             if done:
#                 # env.render()
#                 print(info)
#                 # print(info)
#                 break
#             # agent.get_training_info(
#             #     i, ep_rewards, number_of_iterations, trajectory, starting_position)
#             # self.decay_epsilon(num_games)
#     # return total_rewards

#     # maximum_steps_per_game = 100
#     # total_rewards = np.zeros(num_games)
#     # # env.render()

#     # for i in range(num_games):

#     #     done = False
#     #     ep_rewards = 0
#     #     state = env.reset()
#     #     number_of_iterations = 0
#     #     trajectory = []
#     #     starting_position = env.getAgentPosition()

#     #     while not done:
#     #         number_of_iterations += 1
#     #         if(number_of_iterations == maximum_steps_per_game):
#     #             break

#     #         if(Q_agent):
#     #             action = agent.choose_action(state)
#     #             new_state, reward, done = env.step(action)
#     #             agent.update_q_table(state, new_state, action, reward)

#     #         else:
#     #             action = agent.choose_action()
#     #             new_state, reward, done = env.step(action)

#     #         state = new_state
#     #         trajectory.append(state)
#     #         ep_rewards += reward
#     #     total_rewards[i] = ep_rewards
#     #     if(training):
#     #         agent.get_training_info(i,ep_rewards,number_of_iterations,trajectory,starting_position)

#     # plt.plot(total_rewards)
#     # plt.show()


# if __name__ == "__main__":

#     # env = Grid(7, 7, [(0, 2), (4, 3), (2, 1), (2, 4),(5,6),(0,6)],
#     #            (6, 6), [(0,3),(6,0),(6,4),(1,2)],[(3,5),(5,2),(5,6)])
#     env = Grid(
#         7,
#         7,
#         [(0, 0), (0, 3), (0, 6), (3, 1), (3, 5), (6, 0), (6, 6)],
#         (0, 1),
#         [(1, 1)],
#         [(1, 3), (4, 1), (4, 5)],
#     )
#     env_small = Grid(3,3,[(0,1)],(3,3),[(1,2)],[(1,3)])
#     # print(env.get_state_space())
#     # print(env.grid)
#     # env.step(Actions.EAST)
#     # display_ui(False,env,"Q")

#     # sys.argv.length
#     # Random_agent = RandomAgent(env)
#     # Random_agent.play(70000)
#     arguments = sys.argv[1:]
#     dict_of_possible_arguments = {}
#     if "play" in arguments:
#         num_games = arguments[1]
#         if arguments[2] == "Q":
#             agent = "Q-learning"
#         else:
#             agent = "Sarsa"
#         play(agent, env, num_games, q_table ,500)
#     elif "train" in arguments:
#         agent = arguments[2]
#         num_games = arguments[1]
#         if agent == "Q":
#             Trained_agent_Q = Trained_Agent(env, 1, 0.99, 0.7, "Q_learning", False)
#             # state = env.reset()
#             # done = False
#             # while not done:
#             #     env.print_grid()
#             #     env.step()
#             #     time.sleep(1)
#             # print(Trained_agent_Q.q_table)
#             rewards_Q = Trained_agent_Q.train_agent(num_games,True)
#         elif agent == "S":
#             Trained_agent_S = Trained_Agent(env, 1, 0.99, 0.1, "Sarsa", False)
#             rewards_S = Trained_agent_S.train_agent(num_games,True)
#         else:
#             RandomAgent = RandomAgent(env)
#             rewards_R = RandomAgent.play(num_games)
#     elif "compare" in arguments:
#         if len(arguments) == 1:
#             comparison(env)
#             # im = Image.open(r"comparison_Q_S_R.png")
#             # im.show()
#         else:
#             # num_games = arguments[1]
#             if arguments[2] == "Q":
#                 agent = "Q_learning"
#             else:
#                 agent = "Sarsa"

#             if arguments[1] == "envs":
#                 env_comparison(env_small,env,agent)
#             else:
#                 if arguments[1] == "e":
#                     # image = agent + "_eps_comparison.png"
#                     # im = Image.open(image)
#                     # im.show()
#                     ExploitationVSExploration(env,agent)
#                 elif arguments[1] == "g":
#                     gamma_comparison(env, agent)
#                 else:
#                     alpha_comparison(env, agent)
#     elif "plot" in arguments:
#         if arguments[1] == "Q":
#             im = Image.open(r"Q_learning_plots.png")
#             # This method will show image in any image viewer
#             im.show()
#             # plt.show("Q_learning_plots.png")
#         elif arguments[1] == "S":
#             im = Image.open(r"Sarsa_plots.png")
#             # This method will show image in any image viewer
#             im.show()
#         else:
#             im = Image.open(r"Random_agent_plots.png")
#             im.show()

#     # if len(arguments) > 1:
#     #     if arguments[3] == "Q":
#     #         agent = "Q-learning"
#     #     elif arguments[3] == "S":
#     #         agent = "Sarsa"
#     #     else:
#     #         agent = "Random"
#     #     if arguments[4] == "e":
#     #         ExploitationVSExploration(env,agent)
#     #     elif arguments[4] == "g":
#     #         gamma_comparison(env,agent)
#     #     else:
#     #         alpha_comparison(env,agent)

#     # keywords = ["play", "train", "compare"]
#     # possible_arguments = {}

#     # for i in range(1,len(sys.argv)):
#     #     possible_arguments[i-1] = sys.argv[i]

#     # if len(possible_arguments) == 4

#     # keyword = sys.argv[1]
#     # num_games = sys.argv[2]
#     # algorithm = sys.argv[3]

#     # Trained_agent_Q = Trained_Agent(env,1,0.99,0.1,"Q-learning")
#     # rewards_Q = Trained_agent_Q.train_agent(70000)
#     # Trained_agent_S = Trained_Agent(env,1,0.99,0.1,"Sarsa")
#     # rewards_S = Trained_agent_S.train_agent(70000)
#     # plt.plot(rewards_Q, label='Q Learning')
#     # plt.plot(rewards_S, label='Sarsa')
#     # plt.legend(loc="lower right")
#     # plt.xlabel('Number of Episodes ->')
#     # plt.ylabel('Rewards')
#     # plt.title('Training progress')
#     # plt.show()
#     # play("Q-learning",env,20,500)
#     # play("Sarsa",env,20,500)
#     # comparison(env)
#     # ExploitationVSExploration(env,"Sarsa")
#     # gamma_comparison(env,"Sarsa")
#     # alpha_comparison(env,"Sarsa")

#     # array_for_showing = np.zeros((env.width, env.height,3), dtype=np.uint8)
#     # resizing so we can see our agent in all its glory.
#     # for row in range(0,env.width):
#     #     for col in range(0,env.height):
#     #         if env.grid[row][col] == Grid.AGENT:
#     #             array_for_showing[row][col] = (255, 175, 0)
#     #         elif env.grid[row][col] == Grid.BLOCKED_STATE:
#     #             array_for_showing[row][col] = (0, 0, 0)
#     #         elif env.grid[row][col] == Grid.BOMB:
#     #             array_for_showing[row][col] = (0, 0, 255)
#     #         elif env.grid[row][col] == Grid.GOLD:
#     #             array_for_showing[row][col] = (0, 255, 0)
#     #         else:
#     #             array_for_showing[row][col] = (255, 255, 255)

#     # img = Image.fromarray(array_for_showing, 'RGB')
#     # img = img.resize((300, 300), resample=Image.BOX)
#     # # print(np.array(img))
#     # cv2.imshow("image", np.array(img))
#     # cv2.waitKey()

#     #     # starts an rbg of our size
#     # # env = np.zeros((10,10, 3), dtype=np.uint8)
#     #     # sets the food location tile to green color
#     # # env[food.x][food.y] = d[FOOD_N]
#     # #     # sets the player tile to blue
#     # # env[player.x][player.y] = d[PLAYER_N]
#     # #     # sets the enemy location to red
#     # # env[enemy.x][enemy.y] = d[ENEMY_N]
#     #     # reading to rgb. Apparently. Even tho color definitions are bgr. ???
#     # # env = np.zeros((5, 5))
#     # img = Image.fromarray(array_for_showing, 'RGB')
#     #      # resizing so we can see our agent in all its glory.
#     # img = img.resize((300, 300), resample=Image.BOX)
#     # cv2.imshow("image", np.array(img))
#     # cv2.waitKey(800)  # show it!
#     #    # crummy code to hang at the end if we reach abrupt end for good reasons or not.
#     #    if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
#     #         if cv2.waitKey(50) & 0xFF == ord('q'):
#     #             break
#     #     else:
#     #         if cv2.waitKey(50) & 0xFF == ord('q'):
#     #             break

#     # Q_agent = Q_Agent(env, 1, 0.1, 0.5)
#     # num_trainings = 50000
#     # Q_agent.train_agent(num_trainings)
#     # Randomagent = RandomAgent(env)
#     # num_of_games = 3
#     # max_steps = 100
#     # plt.xlabel("Number of Episodes")
#     # plt.ylabel("Rewards")
#     # plt.plot(play(Q_agent,num_of_games,max_steps))
#     # plt.show()

#     # print(array_for_showing)
#     # img = Image.fromarray(env, "RGB")
#     # img = img.resize((300,300))
#     # cv2.imshow("",np.array(img))
#     # img = Image.fromarray(env, 'RGB')
#     # # resizing so we can see our agent in all its glory.
#     # img = img.resize((300, 300), resample=Image.BOX)
#     # cv2.imshow("image", np.array(img))
#     # env = np.zeros((5, 5), dtype=np.uint8)
#     # print(env)
