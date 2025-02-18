import gymnasium as gym
import matplotlib.pyplot as plt
import random
import torch
from torch import Tensor
from torch import nn


# params for Q-value dueling net
num_inputs = 4      # four input values represent cart's position & velocity, pole's angle & angular velocity, respectively.
num_hidden_a = 256
num_hidden_b = 256
num_outputs = 2     # two output values represent q-values for the two available actions - give the cart a left & right force, respectively.

# training process setup
num_steps = 10 # number of SGD gradient descent steps per epoch.
num_actions = 5000 # number of minimum action-state buffers required to start q-learning process. 
# ! YOU MIGHT WANT TO ADJUST NUM_ACTIONS DURING TRAINING: !
# it is adviced to set a relatively small num_actions at the begining of the training and have it adjusted to a higher value while the average score per game goes high.
num_epoches = 200 # max epoches per learning.
lr = 0.01 # learning rate of SGD gradient descent optimizer for deuling net class.
# ! YOU MIGHT WANT TO ADJUST LR DURING TRAINING: !
# lr should be one of the most frequently adjusted params in q-learning process or in any process that requires a neural net to converge.
# it is adviced to set a relatively large learning rate at the begining of the training when the neural net is far from converge and have it reduced when nn output starts to fluctuate.


# rainbow Q-learning setup

# - classic q-learning params
gamma = 0.95 # importance of future states to the Q-value of previous states. used in DQN loss function below.
terminate_reward = -1 # reward for a terminate action (action that cause game to end before reaches maximum socre of 500).
truncate_reward = 0.1 # reward for a truncate action (last action in a game that has the maximum score of 500).
common_reward = 0.1 # reward for a non-terminate and non_trunacte action.

# - params for noisy net
q_value_dependancy = 0.9 # percentage of actions based on their q-value. a higher dependancy value may lead to higher score per game, however it is not adviced during training process for ramdom actions are needed for the agent to explore more action possibilities. 
noise_std = 0.01 # standard deviation of normally distributed noise given to observation (state). providing observation with a certain amount of noise may lead to the training of a more robust agent.

# - params for prioritized reply
priority_const = 0.1 # importance of states' priority to their q-value. used in DQN loss function below.

# - params for double q-learning
renew_target_net_per_period = 5 # renewing target net with the params of the training net in every renew_target_net_per_period epoch.
# ! YOU MIGHT WANT TO ADJUST RENEW_TARGET_NET_FREQUENCY DURING TRAINING: !
# it is adviced to have a longer renewal period when the net is on its way to converge. however a shorter period is adviced when the params of the net suddenly changes, resulting in utterly different q-values for action-states.

# - params for multi-step
multi_step_size = 3 # the next_state of any target state in q-learning is considered the state after multi_step_size actions the target state. used in DQN loss fucntion below.
# ! YOU MIGHT WANT TO ADJUST MULTI_STEP_SIZE DURING TRAINING: !
# it is adviced to set a relatively smaller multi step size when the game scores are commonly low and few 'next_state' may exist if the param is set too large. however as the game scores increase it is beneficial to set a larger step size so actions with long-term goals may be learned.


# animation setup
animation_speed = 20 # speed of animation being played.


# save high-score agent to file
high_score_threshold = 100 # save agent with average game score above high_score_threshold to file.


# high-loss warning
high_loss_threshold = 1 # raise the training program when facing average DQN loss higher than high_loss_threshold.

# code mode
train_new_net = False # to train a new agent, set train_new_net panel to True. to read trained agent from file and continue training, set train_new_net to False.
show_animation = True # to see animation for trained agent stored in file, set show_animation panel to True. to train agent old or new, set show_animation to False.


class dueling_net:

    """a 5-layer dueling net class disigned by wei jianglan for openai gym's cartpole-v1 mission."""

    def __init__(self, num_inputs: int, num_hidden_a: int, num_hidden_b: int, num_outputs: int, lr: float, requires_grad: bool = True, cuda: bool = False) -> None:
        """initializing a 5-layer dueling net consisted with:
            - input layer, with num_inputs as its size;
            - hidden layer a, with 'ReLU' activation function, and num_hidden_a as its size;
            - hidden layer b, with 'ReLU' activation function, and num_hidden_b as its size;
            - hidden layer c, with its first element the mean of the output layer, and {num_outputs + 1} as its size;
            - output layer, with num_outputs as its size."""
        self.net = nn.Sequential(nn.Linear(in_features=num_inputs, out_features=num_hidden_a),  # input layer -> hidden layer a <in>
                   nn.ReLU(),                                                                   # hidden layer a <in> -> hidden layer a <out>
                   nn.Linear(in_features=num_hidden_a, out_features=num_hidden_b),              # hidden layer a <out> -> hidden layer b <in>
                   nn.ReLU(),                                                                   # hidden layer b <in> -> hidden layer b <out>
                   nn.Linear(in_features=num_hidden_b, out_features=num_outputs+1))             # hidden layer b <out> -> hidden layer c
        self.cuda_ = cuda
        if self.cuda_ is True:
            self.net.cuda()
        if requires_grad is False:
            for params in self.net.parameters():
                params.requires_grad = False
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
        self.net.apply(init_weights)
        self.optimizer = torch.optim.SGD(params=self.net.parameters(), lr=lr)

    def forward(self, inputs: Tensor) -> Tensor:
        """the forward method for dueling net"""
        if self.cuda_: inputs.cuda()
        hidden_layer_c = self.net(inputs.cuda())                                       # value of hidden layer c
        outputs = hidden_layer_c[0] + hidden_layer_c[1:3] - hidden_layer_c[1:3].mean() # value of output layer, hidden_layer_c[0] is the mean of outputs.
        return outputs
    
    def __call__(self, inputs: Tensor) -> Tensor:
        """same as the 'forward' method"""
        return self.forward(inputs)
    
    def step(self) -> None:
        """renew params for the dueling net by SGD gradient descent algorithm"""
        self.optimizer.step()
        self.optimizer.zero_grad()

    def copy(self, other) -> None:
        """renew params by copying params of another dueling_net type object.
        params are only renewed by value, properties such as requires_grad will not be copied."""
        self.net.load_state_dict(other.net.state_dict())
    
    def renew_lr(self, lr: float) -> None:
        """renew learning rate for dueling net"""
        self.optimizer = torch.optim.SGD(params=self.net.parameters(), lr=lr)
        
    def cuda(self):
        """move net to cuda"""
        self.net.cuda()
        self.cuda_ = True
        return self

    def cpu(self):
        """move net to cpu"""
        self.net.cpu()
        self.cuda_ = False
        return self
            
class animator:

    """an animator class designed by wei jianglan."""

    def __init__(self, xlabel: str = None, ylabel: str = None) -> None:
        """initializing animator object.
           - xlabel and ylabel will be shown on the side of two axises. they are optional."""
        self.X = []
        self.Y = []
        self.xlabel = xlabel
        self.ylabel = ylabel
    
    def add(self, x: float, y: float, digit: bool = False):
        """add dot (x, y) to plot and show plot on screen.
           - set 'digit' panel to True to show the y-coordinates on the plot."""
        self.X.append(x)
        self.Y.append(y)
        plt.clf()
        l1, = plt.plot(self.X, self.Y)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend(handles=[l1], labels=[self.ylabel], loc='best')
        if digit:
            for x, y in zip(self.X, self.Y):
                plt.text(x, y, '%.2f' % y, ha = 'center', va = 'bottom')
        plt.pause(0.05)

    def stay(self):
        """let the plot stay on screen until being closed manually."""
        plt.show()


def normalize(state: Tensor) -> Tensor:
    """normalize state values for a better training of deep neural net."""
    return state * torch.tensor([15, 2, 10, 1])



if show_animation: # if code is set to show animation, train_agent and train_new_net should be set False. agent stored in file will be read.
    train_agent = False
    train_new_net = False
else: # code is set to train agent if else. 
    train_agent = True

if train_new_net: # if code is set to train a new agent, new dueling net will be created and new epoch - average_game_score graph will be drawn.
    q_value_net = dueling_net(num_inputs, num_hidden_a, num_hidden_b, num_outputs, lr, cuda=True)
    q_target_net = dueling_net(num_inputs, num_hidden_a, num_hidden_b, num_outputs, lr, requires_grad=False, cuda=True)
    avg_score_grf = animator(xlabel="epoch", ylabel="average score per game")
    epoch = 0
else: # if code is set to further train agent stored in file, agent will be read and previous epoch - average_game_score graph will be restored.
    q_value_net = torch.load('q_value_net.pt', weights_only=False).cuda()
    q_target_net = torch.load('q_target_net.pt', weights_only=False).cuda()
    avg_score_grf = torch.load('avg_score_grf.pt', weights_only=False)
    q_value_net.renew_lr(lr)
    epoch = len(avg_score_grf.X)


# train agent section

if train_agent:
    env = gym.make('CartPole-v1') # obtain gym environment 'CartPole-v1'. 'CartPole-v1' has a max score of 500 per game.
    while epoch < num_epoches:
        print("collecting games...")
        with torch.no_grad():
            if epoch % renew_target_net_per_period == 0: # renew target net
                q_target_net.copy(q_value_net)
            noise = torch.normal(0, noise_std, (num_inputs,)) # add noise to observation to train a more robust agent.
            epoch_buffer = [] # an epoch buffer to record state and action in epoch.
            action_count, game_count = 0, 0 # record action and game count to obtain average game score.
            while action_count < num_actions:
                state = normalize(torch.from_numpy(env.reset()[0])) # obtain initial state.
                terminated, truncated = False, False # determine the game's end.
                game_buffer = [] # a game buffer to record state and action in a single game, later append in epoch_buffer.
                while not terminated and not truncated:
                    if random.uniform(0, 1) < q_value_dependancy:
                        action = q_value_net(state + noise).argmax().item()  # action based on q-value.
                    else:
                        action = random.randint(0, 1) # random action.
                    game_buffer.insert(0, (state, action)) # insert state and action tuple into game_buffer by the reverse order of time.
                    state, reward, terminated, truncated, info = env.step(action) # obtain new state.
                    state = normalize(torch.from_numpy(state)) # normalizing state.
                    action_count += 1 # renew action_count.
                epoch_buffer.append(game_buffer) # append states and actions of a single game into epoch_buffer that records all games in an epoch.
                game_count += 1 # renew game_count.
            avg_score = action_count / game_count # obtain average score per game.
            avg_score_grf.add(epoch, avg_score) # renew epoch - average_score graph.
            print(f"- epoch {epoch} avg score: {avg_score}\n") # print average score in terminal.
            if avg_score > high_score_threshold: # store high-score agent into file.
                torch.save(q_value_net, f'q_value_net_score_{avg_score}.pt')
        print("training q-net...")
        for step in range(num_steps):
            loss = 0 # double DQN loss.
            for game_buffer in epoch_buffer: # traverse game buffers in epoch
                for num_actions_to_last, (state, action) in enumerate(game_buffer): # traverse state & action records in game
                    if num_actions_to_last == 0 and len(game_buffer) < 500: # this state is the terminating state.
                        priority = (q_value_net(state)[action].detach() - terminate_reward) ** 2 * priority_const # obtain state's priority for priority replay.
                        loss += (q_value_net(state)[action] - terminate_reward - priority) ** 2 # double DQN loss function with priority replay.
                    elif num_actions_to_last == 0 and len(game_buffer) == 500: # this state is the truncating state.
                        priority = (q_value_net(state)[action].detach() - truncate_reward) ** 2 * priority_const # obtain state's priority for priority replay.
                        loss += (q_value_net(state)[action] - truncate_reward - priority) ** 2 # double DQN loss function with priority replay.
                    else: # this state is neither the terminating or truncating state.
                        if num_actions_to_last < multi_step_size: # there is less than multi_step_size actions in this game after this state. next_state is set to be the last state in this game.
                            next_state = game_buffer[0][0]
                        else: # next_state is set to be the state after multi_step_size actions of this state.
                            next_state = game_buffer[num_actions_to_last - multi_step_size][0]
                        priority = (q_value_net(state)[action].detach() - q_target_net(next_state)[q_value_net(next_state).argmax()] * gamma - common_reward) ** 2 * priority_const # obtain state's priority for priority replay.
                        loss += (q_value_net(state)[action] - q_target_net(next_state)[q_value_net(next_state).argmax()] * gamma - common_reward - priority) ** 2 # double DQN loss function with priority replay.
            loss /= action_count # obtain average loss.
            if loss > high_loss_threshold: # a high loss may lead to huge SGD step, causing instablity in params. raise exception before taking SGD step.
                raise Exception(f"high loss warning (loss: {loss}). high loss may lead to huge SGD step in gradient descent process.")
            loss.backward() # obtain params' grad.
            q_value_net.step() # renew params for q-value net based on their grads.
        torch.save(q_value_net, 'q_value_net.pt') # save q_value_net to file
        torch.save(q_target_net, 'q_target_net.pt') # save q_target_net to file
        torch.save(avg_score_grf, 'avg_score_grf.pt') # save avg_score_grf to file
        epoch += 1 # renew epoch.


# animation section

if show_animation:
    env = gym.make('CartPole-v1', render_mode='human')
    with torch.no_grad():
        num_actions = 0
        state = normalize(torch.from_numpy(env.reset()[0])) # obtain initial observation.
        terminated, truncated = False, False # determine the game's end.
        while not terminated and not truncated:
            action = q_value_net(state).argmax().item()  # obtain action with maximum q-value.
            state, reward, terminated, truncated, info = env.step(action)
            state = normalize(torch.from_numpy(state)) # obtain new state.
            if num_actions % 20 == 0: # update current score to terminal.
                print(f"{int(num_actions / 500 * 100)} % of game: agent survived {num_actions} actions.")
            num_actions += 1 # renew num_actions (score).
            plt.pause(1 / animation_speed) # pause to show animation graph.
        if num_actions == 500: # game truncated.
            print("game truncated at maximum score 500!")
        else: # game terminated.
            print(f"game terminated at score {num_actions}.")