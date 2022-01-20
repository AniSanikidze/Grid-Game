import numpy as np
import random

class Grid:

    EMPTY = 0
    AGENT = 1
    BOMB = 2
    GOLD = 3

    def __init__(
        self, env_size
    ):
        self.env_size = env_size
        self.grid = np.zeros((env_size, env_size))
        self.bomb_state = self.add_bomb()
        self.gold_state = self.add_gold()
        self.terminal_states = [self.gold_state] + [self.bomb_state]
        self.full_state_space = self.init_state_space()
        self.action_space = [-1, 1, 1, -1]
        self.add_agent_randomly()

    def define_state_space(self):
        state_space = []
        for x in range(self.env_size):
            for y in range(self.env_size):
                state_space.append((x, y))
        state_space = list(set(state_space) - set(self.blocked_states))
        return state_space

    def get_state_space(self):
        return self.full_state_space

    def add_entity_states(self, arr, val):
        arrs = []
        for sub_arr in arr:
            for i, _ in enumerate(sub_arr):
                if sub_arr[i] == 0:
                    arrs.append(sub_arr[:i] + [val] + sub_arr[i + 1 :])
            arrs.append(sub_arr)
        return arrs

    def init_state_space(self):
        empty = [0] * self.env_size * self.env_size
        agents = []
        for i in range(len(empty)):
            agents.append(empty[:i] + [1] + empty[i + 1 :])
        res = self.add_entity_states(self.add_entity_states(agents, 2), 3)
        result = tuple(map(tuple, res))
        obs_dict = {v: k for k, v in enumerate(set(result))}
        return obs_dict

    def get_state(self):
        state = []
        for row in self.grid:
            for y in row:
                state.append(int(y))
        return self.get_state_space()[tuple(state)]

    def get_empty_states(self):
        empty_states = []
        for x in range(self.env_size):
            for y in range(self.env_size):
                if self.grid[x][y] == Grid.EMPTY:
                    empty_states.append((x, y))
        return empty_states

    def action_space_sample(self):
        return random.choice([0, 1, 2, 3])

    def add_agent_randomly(self):
        starting_position_x, starting_position_y = random.choice(
            self.get_empty_states()
        )
        self.grid[starting_position_x][starting_position_y] = Grid.AGENT
        return starting_position_x, starting_position_y

    def add_gold(self):
        starting_position_x, starting_position_y = random.choice(
            self.get_empty_states()
        )
        self.grid[starting_position_x][starting_position_y] = Grid.GOLD
        return starting_position_x, starting_position_y

    def add_bomb(self):
        starting_position_x, starting_position_y = random.choice(
            self.get_empty_states()
        )
        self.grid[starting_position_x][starting_position_y] = Grid.BOMB
        return starting_position_x, starting_position_y

    def is_agent(self, state):
        return self.grid[state[0]][state[1]] == Grid.AGENT

    def is_terminal_state(self, state):
        return state in self.terminal_states

    def is_gold(self, state):
        return self.grid[state[0]][state[1]] == Grid.GOLD

    def is_bomb(self, state):
        return self.grid[state[0]][state[1]] == Grid.BOMB

    def on_grid_move(self, coordinates):
        x = coordinates[0]
        y = coordinates[1]
        return x in range(self.env_size) and y in range(self.env_size)

    def get_agent_position(self):
        for x in range(self.env_size):
            for y in range(self.env_size):
                if self.grid[x][y] == Grid.AGENT:
                    return x, y

    def set_state(self, new_state):
        x, y = self.get_agent_position()
        x_new = new_state[0]
        y_new = new_state[1]
        self.grid[x_new][y_new] = Grid.AGENT
        self.grid[x][y] = Grid.EMPTY

    def print_grid(self):
        for row in self.grid:
            print(row)

    def step(self, action):
        x, y = self.get_agent_position()

        if action == 0:
            x -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y += 1
        elif action == 3:
            y -= 1

        resulting_state = (x, y)

        if not(self.on_grid_move(resulting_state)):
            info = "The agent took an off-grid move"
            return self.get_state(), -10, self.is_terminal_state(resulting_state), info

        else:
            if self.is_gold(resulting_state):
                info = "The agent won the game!"
                self.set_state(resulting_state)
                return self.get_state(), 100, self.is_terminal_state(resulting_state), info
            elif self.is_bomb(resulting_state):
                info = "The agent lost the game!"
                self.set_state(resulting_state)
                return self.get_state(), -100, self.is_terminal_state(resulting_state), info

            else:
                info = "The agent has not reached the termination point"
                self.set_state(resulting_state)
                return self.get_state(), -1, self.is_terminal_state(resulting_state), info

    def reset(self):

        self.grid = np.zeros((self.env_size, self.env_size))
        self.bomb_state = self.add_bomb()
        self.gold_state = self.add_gold()
        self.terminal_states = [self.gold_state] + [self.bomb_state]
        self.add_agent_randomly()
        return self.get_state()

    def render(self):
        print("-------------------------------")
        for row in self.grid:
            for col in row:
                if col == Grid.AGENT:
                    print("A", end="\t")
                elif col == Grid.BLOCKED_STATE:
                    print("X", end="\t")
                elif col == Grid.BOMB:
                    print("B", end="\t")
                elif col == Grid.GOLD:
                    print("G", end="\t")
                else:
                    print("-", end="\t")
            print("\n")
        print("-------------------------------")
