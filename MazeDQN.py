import game
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import game

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    return plt

def calculate_conv_output(W, K, P, S):
    return int((W-K+2*P)/S)+1
    
def calculate_flat_input(dim_1, dim_2, dim_3):
    return int(dim_1*dim_2*dim_3)

class Network(nn.Module): 
    def __init__(self, num_in_positions, num_actions, num_cols, num_rows, batch_size):
        super(Network, self).__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 0)
        self.conv_output_H = calculate_conv_output(num_rows, 3, 0, 1)
        self.conv_output_W = calculate_conv_output(num_cols, 3, 0, 1)
        self.conv_output_H = calculate_conv_output(self.conv_output_H, 3, 0, 1)
        self.conv_output_W = calculate_conv_output(self.conv_output_W, 3, 0, 1)
        self.fc1 = nn.Linear(calculate_flat_input(1, self.conv_output_H, self.conv_output_W)*32, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):  
        x = nn.functional.leaky_relu((self.conv1(x)))
        x = nn.functional.leaky_relu((self.conv2(x)))
        shapes = list(x.size())
        x = x.view(shapes[0], calculate_flat_input(1, self.conv_output_H, self.conv_output_W)*32)
        x = nn.functional.leaky_relu(self.fc1(x))
        x = nn.functional.leaky_relu(self.fc2(x))
        x= self.fc3(x)
        return x

    

class MazeDQN:
    def __init__(self, num_in_positions, num_actions, num_cols, num_rows):
        self.num_actions = num_actions
        self.model = Network(num_in_positions, num_actions, num_cols, num_rows,1)
        self.target_model = Network(num_in_positions, num_actions, num_cols, num_rows, 32)
        learning_rate = 0.0000134
        self.optimizer = optim.Adam(self.model.parameters() ,lr = learning_rate)
        self.criterion = nn.MSELoss()
        self.name = 0   
        self.batch_size = 32 
        self.experience = {'prev_obs' : [], 'a' : [], 'r': [], 'obs' : [], 'done': [] } 
        self.min_exp = 32
        self.max_exp = 256
        self.gamma = 0.93

    def predict(self, inputs):
        x = torch.from_numpy(inputs).float()
        x = x[None,:,:] 
        return self.model(x)
    
    def predict_batch(self, inputs):
        x = torch.from_numpy(inputs).float()
        return self.model(x)

    def target_predict(self, inputs):
        x = torch.from_numpy(inputs).float()
        return self.target_model(x)
    
    def get_action(self, state, epsilon, possible_actions):
        if np.random.random() < epsilon:
            list_possible_actions = list(possible_actions.keys())
            key = np.random.choice(list_possible_actions)
            action = possible_actions[key]
            return action, key
        else:
            prediction = self.predict(np.atleast_2d(state)).detach().numpy()
            list_possible_actions = list(possible_actions.keys())
            for i in range(len(prediction[0])):
                if i+1 not in list_possible_actions:
                    prediction[0][i] = -1e7
            key = np.argmax(prediction)+1
            action = possible_actions[key]
            return action, key

    def add_experience(self, exp):
        if len(self.experience['prev_obs']) >= self.max_exp:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(torch.load(path))
        self.name = path

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def copy_weights(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self):
        if len(self.experience['prev_obs']) < self.min_exp:
            return 0
        
        ids =  np.random.randint(low = 0, high = len(self.experience['prev_obs']), size = self.batch_size)
        states = np.asarray([self.experience['prev_obs'][i] for i in ids])
       
        actions  = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        next_states = np.asarray([self.experience['obs'][i] for i in ids])
        next_states = np.squeeze(next_states)
        
        dones = np.asarray([self.experience['done'][i] for i in ids])

        next_value = np.max(self.target_predict(next_states).detach().numpy(), axis=1)
       
        ''' Q - learning aspect '''
        actual_values = np.where(dones, rewards, rewards+self.gamma*next_value)
      
        

        
        '''  !!!    '''
        actions = np.expand_dims(actions, axis = 1)
        
        

        actions_one_hot = torch.FloatTensor(self.batch_size, self.num_actions).zero_()
        
        actions_one_hot = actions_one_hot.scatter_(1, torch.LongTensor(actions), 1)
        selected_action_values = torch.sum(self.predict_batch(states) * actions_one_hot, dim = 1)
        
        actual_values = torch.FloatTensor(actual_values)
        
        self.optimizer.zero_grad()
        loss = self.criterion(selected_action_values, actual_values)
        loss.backward()
        self.optimizer.step()
        return loss
    

def play_game(DQN):
    maze = game.MazeGame(9,7)
    maze.create_path()
    maze.draw_path()
    maze.render()
    is_finished = False
    reward = 0
    turn = 0
    while not is_finished:
        possible_actions = maze.possible_actions()
        
        action, action_key = DQN.get_action(maze.board, 0.9, possible_actions)
        maze.move(action)
        is_finished, reward = maze.check(turn)
        maze.clean_render()
        turn += 1
    
def generate_data(DQN, min_epsilon, epsilon, copy_step):
    maze = game.MazeGame(9,7)
    maze.create_path()
    maze.draw_path()
    maze.clean_path()
    is_finished = False
    reward = 0
    turn = 0
    decay = 0.9995
    iter = 0
    fails = 0
    repetitions = 1
    loss = 0
    while repetitions % 2 != 0:
        prev_observations = maze.return_state()
        possible_actions = maze.possible_actions()
        action, action_key = DQN.get_action(prev_observations, epsilon, possible_actions)
        maze.move(action)
        is_finished, reward = maze.check(turn)
        observations = maze.return_state()
        turn += 1
        exp = {'prev_obs': prev_observations, 'a' : action_key-1, 'r': reward, 'obs': observations, 'done' : is_finished }
        DQN.add_experience(exp)
        loss += DQN.train()
        if is_finished:
            maze.reset() # TODO
            repetitions +=1
            is_finished = False
        iter +=1
        if iter % 24 == 0:
            epsilon = max(epsilon*decay, min_epsilon)
        if turn % 50 == 0:
            fails = 1
            return fails, turn, loss, epsilon
    return fails, turn, loss, epsilon
    
def dojo(DQN, iterations, min_epsilon, epsilon, copy_step):
    total_loss = 0
    total_fails = 0
    total_turns = 0
    games = 1
    decay = 0.9995
    test_game = game.MazeGame(9,7)
    test_game.load_test_maze("test")
    print(test_game.board)
    test_state = test_game.return_state()
    test_predict = DQN.predict(np.atleast_2d(test_state)).detach().numpy()
    print(test_predict)
    for i in range(iterations):
        fails, turns, loss, epsilon = generate_data(DQN, min_epsilon, epsilon, copy_step)
        total_fails += fails
        total_loss += loss 
        total_turns += turns
        games +=1
        if i % 100 == 0 and i != 0:
            print("total loss:", total_loss)
            print("average turns:", total_turns/ games)
            print("average fails", total_fails / games)
            print("epsilon", epsilon)
            games = 0
            total_fails = 0
            total_loss = 0
            total_turns = 0
            print("games", i)
        if i % 7 == 0:
            DQN.copy_weights()
            test_predict = DQN.predict(np.atleast_2d(test_state)).detach().numpy()
            print(test_predict)
        if i % 10 == 0:
            epsilon = max(epsilon*decay, min_epsilon)
        if i % 1000 == 0 and i != 0:
            plot = plot_grad_flow(DQN.model.named_parameters())
            path = "plot" + str(i)+ ".png"
            plot.savefig(path)
        
    DQN.save_weights("mazenet")


DQN = MazeDQN(24, 4, 9, 7)
dojo(DQN, 10000, 0.15, 0.3, 150)
