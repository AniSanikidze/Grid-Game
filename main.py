from training import Training
from environment import Grid
from play import play
from algorithms_comparison import compare_algorithms
from envs_comparison import env_comparison
from epsilon_comparison import compare_epsilon
from argparser import parse_arguments,check_algorithm,parser,check_decaying_eps,check_env_size

args = parse_arguments()
env_size = check_env_size(args.env_size)
env = Grid(env_size)
epsilon = args.epsilon
gamma = args.gamma
alpha = args.alpha
num_eps = args.num_eps
decaying_eps = check_decaying_eps(args.decaying_eps)
q_table = args.q_table
algorithm = check_algorithm(args.algorithm)
max_steps = args.max_steps
training = Training(env,epsilon,gamma,alpha,algorithm,decaying_eps)

if __name__ == "__main__":
    if args.keyword == "train":
        training.train_agent(num_eps,q_table,max_steps)
    elif args.keyword == "play":
        play(env,training,q_table,num_eps,max_steps)
    elif args.keyword == "compare_algorithms":
        compare_algorithms(env,num_eps,max_steps,epsilon,gamma,alpha,decaying_eps)
    elif args.keyword == "compare_envs":
        env_comparison(algorithm,epsilon,gamma,alpha,decaying_eps,num_eps,max_steps)
    elif args.keyword == "compare_epsilon":
        compare_epsilon(env,algorithm,max_steps,num_eps,gamma,alpha)
    else:
        raise parser.error("Incorrect keyword given. Please type one of the keywords: train, play, compare_algorithms, compare_envs, compare_epsilon")
