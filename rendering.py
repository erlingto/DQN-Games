import numpy as np
import game
import colorgame
import pygame
from time import sleep
import copy
from sys import exit
import MazeDQN
import colorDQN

def render(game):
    if game.type == "maze":
        MazeRendering(game)
    elif game.type == "colorgame":
        ColorRendering(game)

class MazeRendering:
    def __init__(self, maze):
        pygame.init()
        pygame.font.init()
        self.DQN = MazeDQN.MazeDQN(63, 4, 9,7)
        self.DQN.load_weights("mazenet")
        self.side_length = 100

        self.maze = maze
        self.done = False
        self.height = maze.rows
        self.width = maze.col
        self.imagerect = (0, 0)
        self.black = (0,0,0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.open = (125, 125, 125)
        self.blue = (0,0,125)
        self.screen = pygame.display.set_mode([self.side_length*2 + self.width*100, self.side_length + 100* self.height])
        self.text = pygame.font.SysFont('Arial Black', 30)

        while True:
            maze = self.maze
            self.mouse_pos = pygame.mouse.get_pos()
            self.render(self.white, self.black, self.red, self.green, self.white)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.MOUSEBUTTONUP:
                    prev_observations = maze.return_state()
                    possible_actions = maze.possible_actions()
                    action, action_key = self.DQN.get_action(prev_observations, 0, possible_actions)
                    position = maze.game_move(action)
                    print(maze.board)
                    print(action)
                    is_finished, reward = maze.check(0)
                    if is_finished:
                        self.done = True
                        self.render(self.white, self.black, self.red, self.green, self.white)
                        sleep(4)
                        maze = game.MazeGame(9,7)
                        maze.create_path()
                        self.maze = maze
                        self.done = False


    def render(self, background, line, p1_color, p2_color, open_color):
        self.screen.fill(background)
        w = 0
        h = 0
        if not self.done:
            textsurface = self.text.render("Spiller: ", True, (0, 0, 0))
            self.screen.blit(textsurface,(400,0))
        else:
            textsurface = self.text.render("FERDIG!", True, (0, 0, 0))
            self.screen.blit(textsurface,(400,0))
        #vertical lines
        for line in range(self.width+1):
            pygame.draw.line(self.screen, self.black, [line*100 + self.side_length, 50], [line*100 + self.side_length, self.height*100 + 50],3)
        for line in range(self.height+1):   
            pygame.draw.line(self.screen, self.black, [self.side_length, line*100 +50], [self.side_length+self.width * 100, line*100 + 50],3)
        for col in range(self.width):
            for row in range(self.height):
                space = (col, row)
                if self.maze.board[space] == 1:
                    pygame.draw.rect(self.screen, self.blue, (self.side_length + 1 + col* 100, 51 + row*100, 98, 98))
                if self.maze.board[space] == 3:
                    pygame.draw.circle(self.screen, self.green, [self.side_length + col*100 + 50, row * 100 + 100], 45)
                if self.maze.board[space] == 4:
                    pygame.draw.circle(self.screen, self.red, [self.side_length + col*100 + 50, row * 100 + 100], 45)
            w += 1
            if w == self.width:
                    w = 0
                    h+= 1   
        pygame.event.pump()
        pygame.display.flip()   

class ColorRendering:
    def __init__(self, task):
        pygame.init()
        pygame.font.init()
        if task.square_size == 3:
            self.DQN = colorDQN.ColorDQN(9, 9, 3, 3)
            self.DQN.load_weights("colornet5")
        elif task.square_size == 5:
            self.DQN = colorDQN.ColorDQN(25, 25, 5, 5)
            self.DQN.load_weights("colornet55")
        self.side_length = 100

        self.task = task
        self.done = False
        self.height = task.square_size
        self.width = task.square_size
        self.imagerect = (0, 0)
        self.black = (0,0,0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.open = (125, 125, 125)
        self.blue = (0,0,125)
        self.screen = pygame.display.set_mode([self.side_length*2 + self.width*100, self.side_length + 100* self.height])
        self.text = pygame.font.SysFont('Arial Black', 30)

        while True:
            task = self.task
            self.mouse_pos = pygame.mouse.get_pos()
            self.render(self.white, self.black, self.red, self.green, self.white)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.MOUSEBUTTONUP:
                    prev_observations = task.return_state()
                    action = self.DQN.get_action(prev_observations, 0)
                    reward = task.step(action)
                    print(task.board)
                    print(action)
                    
                    if self.task.is_won:
                        self.done = True
                        self.render(self.white, self.black, self.red, self.green, self.white)
                        sleep(4)
                        task = colorgame.ColorGame(self.task.square_size)
                        task.create_task()
                        self.task = task
                        self.done = False


    def render(self, background, line, p1_color, p2_color, open_color):
        self.screen.fill(background)
        w = 0
        h = 0
        if not self.done:
            textsurface = self.text.render("Spiller: ", True, (0, 0, 0))
            self.screen.blit(textsurface,(200,0))
        else:
            textsurface = self.text.render("FERDIG!", True, (0, 0, 0))
            self.screen.blit(textsurface,(200,0))
        #vertical lines
        for line in range(self.width+1):
            pygame.draw.line(self.screen, self.black, [line*300 + self.side_length, 50], [line*300 + self.side_length, self.height*100 + 50],3)
        for line in range(self.height+1):   
            pygame.draw.line(self.screen, self.black, [self.side_length, line*300 +50], [self.side_length+self.width * 100, line*300 + 50],3)
        for col in range(self.width):
            for row in range(self.height):
                space = (col, row)
                if self.task.board[space] == 1:
                    pygame.draw.rect(self.screen, self.blue, (self.side_length + 1 + col* 100, 51 + row*100, 98, 98))
                if self.task.board[space] == -1:
                    pygame.draw.rect(self.screen, self.red, (self.side_length + 1 + col* 100, 51 + row*100, 98, 98))
        pygame.event.pump()
        pygame.display.flip()   


task = colorgame.ColorGame(5)
task.create_task()
render(task)

