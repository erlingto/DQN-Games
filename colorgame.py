import numpy as np 


class ColorGame:
    def __init__(self, square_size):
        self.square_size = square_size
        self.board = np.ones((square_size, square_size), dtype = int)
        self.actions = square_size*square_size
        self.is_won = False
        self.turns = 0
        self.reset_copy = np.copy(self.board)
        self.type = "colorgame"

    def reset(self):
        self.turns = 0
        self.board = np.copy(self.reset_copy)
        self.is_won = False

    def create_task(self):
        steps = 5
        for i in range(steps):
            tile = np.random.randint(0, self.square_size*self.square_size-1)
            self.step(tile)
        self.reset_copy = np.copy(self.board)
        self.turns = 0

    def change(self, position):
        if self.board[position] == 1:
            self.board[position] = -1
        else:
            self.board[position] = 1

    def step(self, action):
        self.turns += 1
        row = action // self.square_size
        if row != 0:
            column = action % (row * self.square_size)
        else:
            column = action

        self.change((row, column))

        if row < self.square_size - 1:
            self.change((row +1, column))
        if row > 0:
            self.change((row -1, column))
        if column < self.square_size - 1:
            self.change((row, column+1))
        if column > 0:
            self.change((row, column-1))

        rewards = self.check()
        return rewards
        
    def check(self):
        check_number = self.board[0][0]
        for i in range(self.square_size):
            for j in range(self.square_size):
                if self.board[i][j] != check_number:
                    self.is_won = False
                    return 0
        self.is_won = True
        return 1
    
    def render(self):
        for i in range(self.square_size):
            for j in range(self.square_size):
                print('|', ' ', sep = '', end = '')
                print(self.board[i][j] ,' ', end = '', sep= '')
            print('| \n', end = '')
            for k in range(self.square_size * 4 +1):
                if k%4 == 0:
                    print('+', end = '')
                else:
                    print('-', end = '')
            print('\n', end = '')

    def return_state(self):
        state = np.copy(self.board)
        return state

