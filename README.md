# DQN-Games
Deep Q-Networks utilized to solve different small/easy games. Games are solved purely by neural network training meaning no search or other methods are used to finish the games.

The motivation behind this repository was to explore different neural network architectures in regards to reinforcement learning. It is mostly a playground for implementing/testing 
different methods in reinforcement learning. There are currently two games in this repository, and they are both fairly simple.

# Maze game
The objective of this game is to move from start to finish in a randomly generated (simple) grid world. The grid can be any size but i have been mostly trying to train the agent in a (9,7) grid. The input to the model is a numpy array of the maze, the array contains information about the location of the agent, the location of the goal and the walls in the maze.  
The model uses deep q learning to maneveur the maze. Deep Q learning is a method in which the model uses a neural network to estimate a value for a given state in the game. It takes as input the game state and gives as output a value estimation of each of the actions it can take. These actions are the different directions the agent can move (up, down, left and right). 

# Color game
The objective of this game is to change every "light" in the grid to the same color. the grid is a (5,5) matrix where every cell can be either on (red) or off (blue). The input to the model is each of the The models action space is the lights in the grid. When one of the lights are chosen the lights in every direction (up, down, left, right) and itself changes color. The game is over when all the lights in the grid is either blue or red. 
