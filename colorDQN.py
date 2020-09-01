import colorgame
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as lines

class Network(nn.Module):
    def __init__(self, num_in_positions, num_actions, num_cols, num_rows, batch_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(num_in_positions, 128)
        self.fc2 = nn.Linear(128, 350, bias = True)
        self.fc3 = nn.Linear(350, 600, bias = True)
        self.fc4 = nn.Linear(600, 350, bias = True)
        self.fc5 = nn.Linear(350, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.fc5(x)
        return x
    
class ColorDQN:
    def __init__(self, num_in_positions, num_actions, num_cols, num_rows):
        self.num_actions = num_actions
        self.num_in_positions = num_in_positions
        self.model = Network(num_in_positions, num_actions, num_cols, num_rows,1)
        self.target_model = Network(num_in_positions, num_actions, num_cols, num_rows, 32)
        learning_rate = 0.000134
        self.optimizer = optim.Adam(self.model.parameters() ,lr = learning_rate)
        self.criterion = nn.MSELoss()
        self.name = 0   
        self.batch_size = 32 
        self.experience = {'prev_obs' : [], 'a' : [], 'r': [], 'obs' : [], 'done': [] } 
        self.min_exp = 100
        self.max_exp = 320
        self.gamma = 0.95

    def predict(self, inputs):
        x = torch.from_numpy(inputs).float().flatten()
        x = x[None,:] 
        return self.model(x)
    
    def predict_batch(self, inputs):
        x = torch.from_numpy(inputs).float().flatten()
        x = x.view(self.batch_size, self.num_in_positions)
        return self.model(x)

    def target_predict(self, inputs):
        x = torch.from_numpy(inputs).float()
        x = x.view(self.batch_size, self.num_in_positions)
        
        return self.target_model(x)

    def get_action(self, state, epsilon): 
        if np.random.random() < epsilon:
            list_possible_actions = [i for i in range(self.num_actions)]
            action = np.random.choice(list_possible_actions)
            return action
        else:
            prediction = self.predict(np.atleast_2d(state)).detach().numpy()
            action = np.argmax(prediction)
            return action

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

def generate_data(DQN, min_epsilon, epsilon, copy_step):
    task = colorgame.ColorGame(5)
    task.create_task()
    
    reward = 0
    turn = 0
    decay = 0.9999
    iter = 0
    loss = 0
    repetitions = 1
    while repetitions % 8 != 0:
        prev_observations = task.return_state()
        action = DQN.get_action(prev_observations, epsilon)
        reward = task.step(action)
        observations = task.return_state()
        exp = {'prev_obs': prev_observations, 'a' : action, 'r': reward, 'obs': observations, 'done' : task.is_won }
        DQN.add_experience(exp)
        if task.turns % 2== 0:
            loss += DQN.train()
        if task.turns == 40:
            epsilon = max(epsilon*decay, min_epsilon)
            return loss, epsilon, task.turns
        if task.is_won:
            repetitions +=1 
            if repetitions % 4 == 0:
                print("WIN")
                return loss, epsilon, task.turns
            task.reset()
            epsilon = max(epsilon*decay, min_epsilon)
    return loss, epsilon, task.turns

def dojo(DQN, iterations, min_epsilon, epsilon, copy_step):
    total_loss = 0
    total_turns = 0
    games = 1
    decay = 0.9999
    task = colorgame.ColorGame(5)
    task.create_task()
    test_task_state = task.return_state()
    for i in range(iterations):
        loss, epsilon, turns = generate_data(DQN, min_epsilon, epsilon, copy_step)
        total_loss += loss 
        total_turns += turns
        games +=1
        if i % 100 == 0 and i != 0:
            print("total loss:", total_loss)
            print("average turns:", total_turns/ games)
            print("epsilon", epsilon)
            games = 0
            total_fails = 0
            total_loss = 0
            total_turns = 0
            print("games", i)
        if i % 10 == 0:
            print(i)
            print(DQN.predict(test_task_state).detach())
        if i % 10 == 0:
            DQN.copy_weights()
        epsilon = max(epsilon*decay, min_epsilon)
        """
        if i % 1000 == 0 and i != 0:
            plot = plot_grad_flow(DQN.model.named_parameters())
            path = "ColorPlot" + str(i)+ ".png"
            plot.savefig(path)
        """          
    DQN.save_weights("colornetminRelu")


DQN  = ColorDQN(25, 25, 5, 5)

dojo(DQN, 50000, 0.2, 0.9, 32)
