import gymnasium as gym
import matplotlib.pyplot as plt
import math
import random
import torch
from torch import Tensor
from animator import animator

class softmax_net:
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        self.w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)
    
    def forward(self, X: Tensor) -> Tensor:
        return torch.squeeze((X.reshape((-1, self.w.shape[0])) @ self.w + self.b).softmax(dim=1))
    
    def step(self, lr):
        with torch.no_grad():
            for param in [self.w, self.b]:
                param += param.grad * lr
                param.grad.zero_()


def advantage_function_with_bias(score, t, bias):
    if score <= retard_decrease:
        return advantage_const * math.exp(math.log(time_const) * (t + retard_decrease - score)) - bias
    else:
        if t < score - retard_decrease:
            return advantage_const - bias
        else:
            return advantage_const * math.exp(math.log(time_const) * (t + retard_decrease - score)) - bias

def get_advanteges_sum(score):
    if score <= retard_decrease:
        return retard_advantage_sum - advantage_const * (math.exp(math.log(time_const) * (retard_decrease - score)) - 1) / (time_const - 1)
    else:
        return advantage_const * (score - retard_decrease) + retard_advantage_sum

# hyperparams
time_const = 0.95
retard_decrease = 50
advantage_const = 100
num_epoches = 1000
num_steps = 50
lr = 2e-6
num_inputs = 4
num_outputs = 2
retard_advantage_sum = advantage_const * (math.exp(math.log(time_const) * retard_decrease) - 1) / (time_const - 1)

net = softmax_net(num_inputs, num_outputs)

grf = animator(xlabel="step", ylabel="average score per epoch")

env = gym.make('CartPole-v1')

# train net
for step in range(num_steps):
    with torch.no_grad():
        records = []
        total_advantage_sum = 0
        total_score = 0
        for epoch in range(num_epoches):
            state = env.reset()
            X = torch.from_numpy(state[0])
            epoch_score = 0
            terminated = False
            truncated = False
            epoch_record = []
            while not terminated and not truncated:
                random_float = random.uniform(0, 1)
                if net.forward(X)[0] > random_float:    action = 0  # left force
                else:                                   action = 1  # right force
                epoch_record.append((X, action))
                observation, reward, terminated, truncated, info = env.step(action)
                X = torch.from_numpy(observation)
                epoch_score += reward
            if truncated:
                print("truncated")
            total_advantage_sum += get_advanteges_sum(score=epoch_score)
            total_score += epoch_score
            records.append((epoch_record, epoch_score))
    advantage_function_bias = total_advantage_sum / total_score
    ppo_value = torch.zeros(1)
    for [epoch_record, epoch_score] in records:
        for t, [X, action] in enumerate(epoch_record):
            ppo_value += advantage_function_with_bias(score=epoch_score, t=t, bias=advantage_function_bias) * torch.log(net.forward(X)[action])
    ppo_value.backward()
    net.step(lr=lr)
    grf.add(x=step, y=total_score/num_epoches, text=False)

grf.stay()


# see animation
with torch.no_grad():
    env = gym.make('CartPole-v1', render_mode='human')
    state = env.reset()
    X = torch.from_numpy(state[0])
    epoch_score = 0
    terminated = False
    truncated = False
    while not terminated and not truncated:
        random_float = random.uniform(0, 1)
        if net.forward(X)[0] > random_float:    action = 0  # left force
        else:                                   action = 1  # right force
        observation, reward, terminated, truncated, info = env.step(action)
        plt.pause(0.1)
        X = torch.from_numpy(observation)
        epoch_score += reward