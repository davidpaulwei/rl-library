import gymnasium as gym
import torch
import math
import random
import matplotlib.pyplot as plt
from torch import Tensor
from numpy import ndarray
import numpy
from animator import animator2


class Grids2d:

    def __init__(self, xrange: tuple, yrange: tuple, num_xgrids: int, num_ygrids: int, num_values: int) -> None:
        self.values = numpy.zeros((num_xgrids, num_ygrids, num_values))
        self.add_count = numpy.zeros((num_xgrids, num_ygrids, num_values))
        self.xrange = numpy.array(xrange)
        self.yrange = numpy.array(yrange)
        self.num_xgrids = num_xgrids
        self.num_ygrids = num_ygrids
        self.num_values = num_values
        self.grid_xwidth = (xrange[1] - xrange[0]) / num_xgrids
        self.grid_ywidth = (yrange[1] - yrange[0]) / num_ygrids

    def __iadd__(self, other: ndarray):
        self.values += other
        self.add_count += 1
        return self

    def add(self, position: ndarray, value_index: int, value: float) -> None:
        grid_xindex = int((position[0] - self.xrange[0]) // self.grid_xwidth)
        grid_yindex = int((position[1] - self.yrange[0]) // self.grid_ywidth)
        self.values[grid_xindex][grid_yindex][value_index] += value
        self.add_count[grid_xindex][grid_yindex][value_index] += 1

    def get_values(self, position: ndarray) -> ndarray:
        grid_xindex = int((position[0] - self.xrange[0]) // self.grid_xwidth)
        grid_yindex = int((position[1] - self.yrange[0]) // self.grid_ywidth)
        return self.values[grid_xindex][grid_yindex]

    def average(self) -> ndarray:
        for x, mat in enumerate(self.add_count):
            for y, vec in enumerate(mat):
                for i, val in enumerate(vec):
                    if val == 0:
                        self.add_count[x][y][i] = 1
        return self.values / self.add_count
    
    def reset(self) -> None:
        self.values = numpy.zeros(self.num_xgrids, self.num_ygrids, self.num_values)
        self.add_count = numpy.zeros(self.num_xgrids, self.num_ygrids, self.num_values)
        


def get_action(state: ndarray, Q_value: Grids2d) -> int:
    random_float = random.uniform(0, 1)
    if random_float <= random_action_ratio: # return random action
        return random.randint(0, 2)
    else:   # return action based on action-utility function
        return Q_value.get_values(state).argmax()

def get_reward(this_state: ndarray, next_state: ndarray) -> float:
    return next_state[1] ** 2 - this_state[1] ** 2


# hyperparams
time_const = 0.5
random_action_ratio = 0.1
num_epoches = 200
num_steps = 10
num_xgrid = 20
num_ygrid = 10
final_reward = 1e-3
lr = 0.1

# env params
pos_range = (-1.2, 0.6)
vel_range = (-0.07, 0.07)
num_actions = 3

env = gym.make('MountainCar-v0')

grf = animator2(xlabel="step", y1label="average max position", y2label="success rate")

Q_function = Grids2d(xrange=pos_range, yrange=vel_range, num_xgrids=num_xgrid, num_ygrids=num_ygrid, num_values=num_actions)

Q_target_function = Grids2d(xrange=pos_range, yrange=vel_range, num_xgrids=num_xgrid, num_ygrids=num_ygrid, num_values=num_actions)

# train Q function
for step in range(num_steps):
    max_pos_sum = 0
    success_count = 0
    for epoch in range(num_epoches):
        state = env.reset()[0]
        terminated = False
        truncated = False
        max_pos = -1.2
        while not terminated and not truncated:
            action = get_action(state, Q_function)
            prev_state = state
            state, reward, terminated, truncated, info = env.step(action)
            if terminated: # car arrived at flag:
                Q_target_function.add(prev_state, action, final_reward)
                success_count += 1
            else: # car not arrived at flag
                Q_target_function.add(prev_state, action, get_reward(prev_state, state) + time_const * Q_function.get_values(state).max())
            if state[0] > max_pos:
                max_pos = state[0]
        max_pos_sum += max_pos
    Q_function += lr * (Q_target_function.average() - Q_function.values)
    grf.add(x=step, y1=max_pos_sum/num_epoches, y2=success_count/num_epoches)
grf.stay()

# see animation
random_action_ratio = 0
env = gym.make('MountainCar-v0', render_mode='human')
state = env.reset()[0]
terminated = False
truncated = False
while not terminated and not truncated:
    action = get_action(state, Q_function)
    state, reward, terminated, truncated, info = env.step(action)
    plt.pause(0.05)
