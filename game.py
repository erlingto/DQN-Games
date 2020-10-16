import numpy as np

class MazeGame:
    def __init__(self, num_col, num_rows):
        self.col = num_col
        self.rows = num_rows
        self.board = np.zeros((num_col, num_rows), dtype = int)
        #self.board = np.zeros(num_col*num_rows, dtype = int)
        self.num_walls = 20
        self.actions = {1: 'up', 2: 'down', 3: 'left', 4: 'right'}
        self.position = None, None
        self.start = None, None
        self.end = None, None
        self.type = "maze"
        
    def clean_render(self):
        clean_board = self.board.copy()
        for i in range(len(self.board)):
            if clean_board[i] == 10:
                clean_board[i] = 0

        clean_board[self.position] = 8

        for k in range(self.col * 4 +1):
                if k%4 == 0:
                    print('+', end = '')
                else:
                    print('-', end = '')
        print('\n', end = '')
        for i in range(self.rows):
            for j in range(self.col):
                print('|', ' ', sep = '', end = '')
                print(clean_board[i*self.col + j] ,' ', end = '', sep= '')
            print('| \n', end = '')
            for k in range(self.col * 4 +1):
                if k%4 == 0:
                    print('+', end = '')
                else:
                    print('-', end = '')
            print('\n', end = '')

    def render(self):
        for k in range(self.col * 4 +1):
                if k%4 == 0:
                    print('+', end = '')
                else:
                    print('-', end = '')
        print('\n', end = '')
        for i in range(self.rows):
            for j in range(self.col):
                print('|', ' ', sep = '', end = '')
                print(self.board[i*self.col + j] ,' ', end = '', sep= '')
            print('| \n', end = '')
            for k in range(self.col * 4 +1):
                if k%4 == 0:
                    print('+', end = '')
                else:
                    print('-', end = '')
            print('\n', end = '')


    def possible_actions(self):
        position = self.position
        column = position[0]
        row = position[1]
        actions = self.actions.copy()
        if row == self.rows-1: 
            actions.pop(2)
        elif self.board[column][row+1] == 1:
            actions.pop(2)
        if row == 0: 
            actions.pop(1)
        elif self.board[column][row-1] == 1:
            actions.pop(1)
        if column == self.col-1:
            actions.pop(4)
        elif self.board[column+1][row] == 1:
            actions.pop(4)
        if column == 0:
            actions.pop(3)
        elif self.board[column-1][row] == 1:
            actions.pop(3)
        
        return actions

    def remove_opposite_action(self, list_of_actions, action, action2):
        if action == "up" or action2 == "up":
            list_of_actions.pop(2, None)
        elif action == "down" or action2 == "down":
            list_of_actions.pop(1, None)
        elif action == "right" or action2 == "right":
            list_of_actions.pop(4, None)
        elif action == "left" or action2 == "left":
            list_of_actions.pop(3, None)
        return list_of_actions
    
    def move(self, action):
        position = self.position
        if action == "up":
            position = (position[0],position[1]-1)
        elif action == "down":
            position = (position[0],position[1]+1)
        elif action == "left":
            position = (position[0]-1,position[1])
        elif action == "right":
            position = (position[0]+1,position[1])
        self.board[self.position] = 0
        self.position = position
        return position 

    def game_move(self, action):
        position = self.position
        if action == "up":
            position = (position[0],position[1]-1)
        elif action == "down":
            position = (position[0],position[1]+1)
        elif action == "left":
            position = (position[0]-1,position[1])
        elif action == "right":
            position = (position[0]+1,position[1])
        self.board[self.position] = 0
        self.position = position
        self.board[self.position] = 3
        return position 

    def check(self, turn):
        finished = False
        if self.position == self.end:
            finished = True
            return finished, 1
        else:
            return finished, 0
    
<<<<<<< HEAD
    def create_path(self, steps):
        locations = {i : 0 for i in range(self.col*self.rows)}
        i = 0
        for x in range(self.col):
            for y in range(self.rows):
                locations[i] = (x,y)
                i+=1
        
        start_key = locations[np.random.choice(locations.keys())]
        locations.pop(start_key)
        self.path = [0] * (steps + 1)
        self.path[0] = self.start
        self.position = self.start
        for i in range(steps):
            action = possible_actions[np.random.choice(list(possible_actions.keys()))]
            self.position = self.move(action)
            self.path[i+1] = self.position
            locations.pop(self.position)
    
    """
=======
    def create_path(self, steps): 
        locations = {i : (0,0) for i in range(self.col*self.rows)}
        i = 0
        for x in range (self.col):
            for y in range(self.rows):
                locations[i] = (x, y)
                i+= 1
        start_key = np.random.choice(list(locations.keys()))
        self.start = locations[start_key]
        self.position = self.start
        self.board[self.start] = 4
        self.path = [0] * (steps + 1)
        self.path[0] = self.position
        for i in range(steps):
            possible_actions = self.possible_actions()
            action = possible_actions[np.random.choice(list(possible_actions.keys()))]
            self.position = self.move(action)
            self.path[i+1] = self.position
            
            if action in self.possible_actions():
                self.position = self.move(action)
                self.path[i + 1] = self.position
                i+= 1
                if action in self.possible_actions():
                    self.position = self.move(action)
                    self.path[i + 1] = self.position
                    i+= 1
        longest = -1
        temp_end = None
        for i in range(len(self.path)):
            distance = abs(self.path[i][0]-self.start[0]) + abs(self.path[i][1]-self.start[1])
            if (distance > longest ):
                longest = distance
                temp_end = self.path[i]
                
        self.end = temp_end
        self.position = self.start
        self.board[self.end] = 6
        for i in range(len(locations.keys())):
            if locations[i] in self.path:
                locations.pop(i)

        for i in range(self.num_walls):
            choice = np.random.choice(list(locations.keys()))
            if locations[choice] not in self.path:
                self.board[locations[choice]] = 1
            locations.pop(choice)
    """            
>>>>>>> b6207beb443f0d5b9344f18876d7d943c62d05eb
    def create_path(self):
        list_of_locations = [0] * (self.col*self.rows)
        list_of_locations_keys = [i for i in range(self.col*self.rows)]
        k = 0
        for i in range (self.col):
            for j in range(self.rows):
                list_of_locations[k] = (i, j)
                k+=1
        start_key = np.random.choice(list_of_locations_keys)     
        self.start = list_of_locations[start_key]
        self.board[self.start] = 3

        list_of_locations_keys.remove(start_key)
        list_steps = [14, 16, 20 ,24]
        steps = np.random.choice(list_steps)
        self.path = [0] * (steps + 1)
        self.path[0] = self.start
        self.position = self.start
        possible_actions = self.possible_actions()
        action2 = 0
        for i in range(steps):
            action = possible_actions[np.random.choice(list(possible_actions.keys()))]
            self.position = self.move(action)
            self.path[i+1] = self.position
            pos_key = self.position[1]*self.col+self.position[0]
            if (pos_key in list_of_locations_keys):
                list_of_locations_keys.remove(pos_key)
            possible_actions = self.possible_actions()
            action2 = action
            possible_actions = self.remove_opposite_action(possible_actions, action, action2)
        longest = 0    
        temp_end = None
        for i in range(len(self.path)):
            distance = abs(self.path[i][0]-self.start[0]) + abs(self.path[i][1]-self.start[1])
            if (distance > longest ):
                longest = distance
                temp_end = self.path[i]
        self.end = temp_end
        end_key = self.end[1]*self.col+self.end[0]
        if (end_key in list_of_locations_keys):
            list_of_locations_keys.remove(end_key)
        self.board[self.end] = 4
        self.position = self.start
        for i in range(self.num_walls):
            choice = np.random.choice(list_of_locations_keys)
            list_of_locations_keys.remove(choice)
            if list_of_locations[choice] not in self.path:
                self.board[list_of_locations[choice]] = 1
<<<<<<< HEAD
    """
    
=======

    """
>>>>>>> b6207beb443f0d5b9344f18876d7d943c62d05eb
    def draw_path(self):
        for i in range(len(self.path)-1):
            self.board[self.path[i+1]] = 10
        self.board[self.start] = 3
        self.board[self.end] = 4

    def clean_path(self):
        for i in range(len(self.path)-1):
            self.board[self.path[i+1]] = 0
        self.board[self.end] = 4
        self.board[self.start] = 3

    def reset(self):
        self.board[self.position] = 4
        self.position = self.start
        self.board[self.position] = 3
<<<<<<< HEAD
        
=======
    
    def return_state_1d(self):
        state = np.copy(self.board)
        state[self.position] = 3
        state[self.end] = 4
        return state

    def return_state_2d(self):
        state = [None] * 2
        position_end = np.zeros((self.col, self.rows), dtype = int)
        walls = np.zeros((self.col, self.rows), dtype = int)
        for i in range(self.col):
            for j in range(self.rows):
                if self.board[i][j] == 1:
                    walls[i][j] = 1
        position_end[self.position] = 1
        position_end[self.end] = 2
        state[0] = position_end
        state[1] = walls    
        return state


>>>>>>> b6207beb443f0d5b9344f18876d7d943c62d05eb
    def return_state(self):
        board = self.board
        state = [None] * 3
        position = np.zeros((self.col, self.rows), dtype = int)
        walls = np.zeros((self.col, self.rows), dtype = int)
        end = np.zeros((self.col, self.rows), dtype = int)
        for i in range(self.col):
            for j in range(self.rows):
                if board[i][j] == 1:
                    walls[i][j] = 1
<<<<<<< HEAD

=======
>>>>>>> b6207beb443f0d5b9344f18876d7d943c62d05eb
        position[self.position] = 1
        end[self.end] = 1
        state[0] = position
        state[1] = end
        state[2] = walls
        return state


    def create_test_maze(self, path):
        f = open(path+".txt", "w")
        f.write(str(self.board))
        f.write("\n")
        f.write(str(self.start))
        f.write("\n")
        f.write(str(self.end))

    def load_test_maze(self, path):
        f = open(path+".txt", "r")
        content = f.readlines()
        end = content[len(content)-1] 
        start = content[len(content)-2]
        content.remove(end)
        content.remove(start)
        self.end = (int(end[1]), int(end[4]))
        self.start = (int(start[1]), int(start[4]))
        self.position = self.start
        self.board = np.zeros((self.col, self.rows), dtype = int)
        i = 0
        for line in content:
            line = line.replace(" ", "")
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.strip()
            self.board[i] = np.fromstring(line, np.int8) - 48
            i+=1
        self.board = np.array(self.board)


game = MazeGame(9,7)
game.create_path(16)
game.draw_path()
game.clean_path()

game.create_test_maze("test")

