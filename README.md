# Grid-Game
![GridGameVisualization](https://user-images.githubusercontent.com/56120787/150558179-b69a780f-0e03-408c-b564-28b53c19418e.gif)

The grid game is a prototype similar to OpenAI gym environments, on which  reinforcement learning algorithms, namely Q-learning and Sarsa are applied. The prototype allows users to train the agent using the chosen algorithm, test the trained agent by playing the game, and compare algorithms under different sized environments and hyperparameters.

## Installation

```bash

git clone https://github.com/AniSanikidze/Grid-Game

cd Grid-Game

pip install -r requirements.txt

```

## Usage
The prototype  runs based on the given command line arguments. The user needs to enter one of the following keywords (train, play, compare_algorithms, compare_envs, compare_epsilon) followed by the number of episodes. The optional agruments that the user can give are the following:

```bash

--algorithm (default: Q)

--q_table (By default the Q table is not specified)

--env_size (default: 7)

--max_steps (default: 100)

--epsilon (default: 1)

--gamma (default: 0.9)

--alpha (default: 0.5)

--decaying_eps (default: True)

```


### Training
To train the agent, the user needs to enter the keyword 'train' followed by the number of episodes. The default algorithm is Q-learning; therefore, if no other arguments are provided, the agent will be trained with Q-learning for the given number of episodes.

Training without optional arguments

```bash

python main.py train 300000

```
Training with optional arguments

```bash

python main.py train 300000 --algorithm S --q_table Example_Q_Table_Sarsa --env_size 7  --epsilon 1 --gamma 0.9 --alpha 0.5 --decaying_eps True --max_steps 100

```

If the name for Q table is provided, then it will be saved in the q_tables folder, otherwise the Q table will not be saved.

### Play

To check how the trained agent plays the grid game, the user needs to enter the kewyord 'play' followed by the number of episodes. Also, for playing the game user needs to provide the name of the saved Q table.

Playing the game

```bash

python main.py play 10 --q_table Q_table_Q_learning

```
Playing the game with optional arguments

```bash

python main.py play 10 --q_table Q_table_Q_learning --max_steps 100 --env_size 7

```

The given env_size and the environment size in which the agent was trained should correspond to each other.

The user can test the game with already provided Q tables: Q_tabke_Q_learning and Q_table_Sarsa. The Q tables were saved after training the agent for 500 000 episodes. Both of the saved Q tables are provided in the q_tables folder.


### Comparisons
The users can compare different sized grids and with different learning policies. The comparisons are saved in the plots folder.


#### Comparison of different sized environments
Comparing algorithms without optional arguments

```bash

python main.py compare_algorithms 150000

```
Comparing algorithms with optional arguments

```bash

python main.py compare_algorithms 150000 --env_size 7 --epsilon 1 --gamma 0.9 --alpha 0.5 --max_steps 100 --decaying_eps True

```

After each algorithm finishes the training, the plots are shown. The user will need to close the plots for the comparison process to be continued.

#### Comparison of 3x3 and 7x7 environments

The users can compare how the chosen algorithm works in the 3 by 3 and 7 by 7 grids. Without specifying the algorithm, Q-learning will be applied to the environments.

```bash

python main.py compare_envs 150000

```

Comparing environments with all possible optional arguments

```bash

python main.py compare_envs 150000 --algorithm S --eps 1 --gamma 0.9 --alpha 0.5 --decaying_eps True --max_steps 100

```

#### Comparison of algorithm's performance with different epsilon values

```bash

python main.py compare_epsilon 150000

```
Comparison with optional arguments
```bash

python main.py compare_epsilon 150000 --env_size 7 --max_steps 100 --algorithm S

```
