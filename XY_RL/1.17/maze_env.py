"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk    # GUI

UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width
START_POINT = np.array([0.5*MAZE_W*UNIT, 0.5*MAZE_H*UNIT])  # origin
TARGET_POINT = np.array([0.9*MAZE_W*UNIT, 0.5*MAZE_H*UNIT])

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['l', 'r', 'n'] # left, right, do nothing
        self.n_actions = len(self.action_space)
        self.n_features = 2 # type_uav d env type_mis
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))   # the shape of figure
        self._build_maze()

    def _build_maze(self):  # GUI, wait changing!!!!!!!!!!!!!!!!!!!!!!!!!!1
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create line
        self.canvas.create_line(0, 0.5*MAZE_H*UNIT, MAZE_W*UNIT, 0.5*MAZE_H*UNIT)

        origin = START_POINT

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 2, origin[1] - 2,
            origin[0] + 2, origin[1] + 2,
            fill='red')

        # create target
        oval_center = TARGET_POINT
        self.oval = self.canvas.create_oval(
            oval_center[0] - 2, oval_center[1] - 2,
            oval_center[0] + 2, oval_center[1] + 2,
            fill='yellow'
        )

        # pack all
        self.canvas.pack()


    def reset(self):
        self.update()
        time.sleep(0.1) # every time reset will wait with this time
        self.canvas.delete(self.rect)
        origin = START_POINT
        self.rect = self.canvas.create_rectangle(   # reset to the origin point
            origin[0] - 2, origin[1] - 2,
            origin[0] + 2, origin[1] + 2,
            fill='red')
        # return observation
        # there should be one dim????????????????!!!!!!!!!!!!!!!!!!!!!!
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        # provide go out of the canvas
        if action == 0:   # left
            if s[0] > 0.1*UNIT:
                base_action[0] -= 0.1*UNIT
        elif action == 1:   # right
            if s[0] < (MAZE_H - 0.1) *UNIT:
                base_action[0] += 0.1*UNIT
        elif action == 2:   # do nothing
            pass

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.rect)  # next state

        # reward function
        if next_coords == self.canvas.coords(self.oval):
            reward = 50
            done = True
        else:
            reward = -1
            done = False
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()


