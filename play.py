from tkinter.tix import Tree
from UI import display_ui
import pickle

def load(q_table):
    path = "q_tables\{}.pkl".format(q_table)
    with open(path, "rb") as G:
        try:
            return pickle.load(G)
        except FileNotFoundError as e:
            print(e)

def play(env,trained_agent, q_table_to_load, num_games, maximum_steps_per_game):

    q_table = load(q_table_to_load)

    for i in range(1, num_games + 1):
        done = False
        state = env.reset()
        number_of_iterations = 0
        print(f"****EPISODE {i}****")

        while number_of_iterations != maximum_steps_per_game:
            number_of_iterations += 1
            action = trained_agent.max_action(state,q_table)

            if action == 0:
                print("North")
            elif action == 1:
                print("South")
            elif action == 3:
                print("West")
            elif action == 2:
                print("East")

            new_state,rewards, done, info = env.step(action)
            state = new_state
            display_ui(done, env)

            if done:
                print(info)
                break
