import argparse
from ast import parse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('keyword', type=str,
                    help='enter one of the keywords: train, play, compare_algorithms, compare_envs, compare_epsilon')
parser.add_argument('num_eps', type=int,
                    help='enter the number of episodes')
parser.add_argument('--algorithm', type=str, default="Q",
                    help='enter the chosen algorithm Q for Q-learning or S for Sarsa, (example: %(default)s)')
parser.add_argument('--q_table', type=str, default="",
                    help='enter the name of the saved Q-table (example: Q_learning_Q-table.pkl')
parser.add_argument('--env_size', type=int, default=7,
                    help='enter the size of the grid (default: %(default)s) Note: when playing the game, the environment size should correspond to the size used while training the agent.')
parser.add_argument('--epsilon', type=float, default=1,
                    help='enter the value of starting epsilon (default: %(default)s)')
parser.add_argument('--decaying_eps', default='True',
                    help='enter the True for e-greedy policy, otherwise the epsilon value will be fixed (default: %(default)s)')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='enter the value of gamma (default: %(default)s)')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='enter the value of alpha (default: %(default)s)')
parser.add_argument('--max_steps', type=int, default=100,
                    help='enter the maximum number of steps that agent can take per episode (default: %(default)s)')

def check_algorithm(algorithm):
    if algorithm == "Q":
        return "Q-learning"
    elif algorithm == "S":
        return "Sarsa"
    else:
        parser.error("Incorrect algorithm - " + algorithm +
                     " given. Please enter Q for Q-learning or S for Sarsa")

def check_decaying_eps(decaying_eps):
    if decaying_eps != 'True' and decaying_eps != 'False':
        parser.error("Incorrect argument given for --decaying_eps. The argument should be either True or False")
    else:
        if decaying_eps == "True":
            return True
        else:
            return False
def check_env_size(env_size):
    if env_size < 2:
        parse.error("The enviornment size should be at least 2.")
    else:
        return env_size

def parse_arguments():
    args = parser.parse_args()
    return args



