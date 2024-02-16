import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random
import math
import torch.nn.functional as F
from itertools import count

# Define the neural network model


class DQN(nn.Module):
    def __init__(self, n_obs, n_act):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_obs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.out = nn.Linear(256, n_act)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # x = torch.softmax(self.out(x), dim=-1)
        return self.out(x)


"""class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)"""


"""def discount_reward(rewards, gamma):
    running_reward = 0
    result = np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        result[i] = rewards[i] + gamma * running_reward
        running_reward += rewards[i]
    return result"""


"""Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'eps'))"""
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def pull(self, batch_size):
        out = list(self.memory)
        self.memory = deque([], maxlen=self.memory.maxlen)
        return out

    def __len__(self):
        return len(self.memory)


"""# Initialize environment and model
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")
state, info = env.reset()
n_act = env.action_space.n
# Obs: Cart position / Cart velocity / pole angle / pole angular velocity
n_obs = len(state)

model = DQN(n_obs, n_act)
optimizer = optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
loss_fn = nn.CrossEntropyLoss()
memory = ReplayMemory(10000)

# Training loop
n_episodes = 30000
max_time = 200
all_rewards = []
train_data = deque(maxlen=10000)  # Use deque to keep the last N experiences
gamma = 0.99
batch_size = 128

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
steps_done = 0"""
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
max_time = 200

# Get number of actions from gym action space
env = gym.make("CartPole-v1")
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # print("state_batch", state_batch)
    # print("reward_batch", reward_batch)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 300

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32,
                         device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            # episode_durations.append(t + 1)
            # plot_durations()
            print("episode duration :", t)
            break


"""for i_episode in range(n_episodes):
    obs, _ = env.reset()
    state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    episode_reward = 0
    episode_transitions = []

    for t in range(max_time):
        # Forward pass to get action probabilities
        q_values = model(state).squeeze()  # remove dimension of size 1
        action_probabilities = torch.softmax(q_values, dim=0)
        action = torch.distributions.Categorical(action_probabilities).sample()
        # fit to 2D tensor, needed for further tensor concatenation
        action = action.view(1, 1
        action = select_action(state)

        # Take a step in the environment
        obs, reward, done, _, _ = env.step(action.item())
        next_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor([reward])

        # Store the transition
        episode_transitions.append((state, action, next_state, reward))
        state = next_state
        episode_reward += reward.item()

        if done:
            all_rewards.append(episode_reward)
            # print(f"Episode {i_episode} done with reward: {episode_reward}")

            rewards = np.array([transition[3].item()
                               for transition in episode_transitions])
            # discounted_rewards = discount_reward(rewards, gamma)
            discounted_rewards = rewards

            # Update memory
            for idx, (state, action, next_state, _) in enumerate(episode_transitions):
                memory.push(state, action, next_state, torch.tensor(
                    [discounted_rewards[idx]], dtype=torch.float32), i_episode)

            # Every batch_size episodes, perform a training step using all transition from memory
            if (i_episode + 1) % batch_size == 0 and i_episode != 0:

                # transitions = memory.pull(batch_size)
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))
                # print('batch \n', batch)
                # print([s.shape for s in batch.state])

                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                next_state_batch = torch.cat(batch.next_state)

                # print("state_batch", state_batch)
                # print("reward_batch", reward_batch)

                current_q_values = model(state_batch).gather(1, action_batch)
                # max_next_q_values = model(
                #    next_state_batch).detach().max(1)[0]
                # expected_q_values = reward_batch + (gamma * max_next_q_values
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                        batch.next_state)), dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                   if s is not None])
                next_state_values = torch.zeros(batch_size)
                with torch.no_grad():
                    next_state_values[non_final_mask] = model(
                        non_final_next_states).max(1).values
                # Compute the expected Q values
                expected_q_values = (next_state_values * gamma) + reward_batch
                # print("current_q_values", current_q_values)
                # print("expected_q_values", expected_q_values.unsqueeze(1))
                loss = nn.SmoothL1Loss()(current_q_values, expected_q_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            break

    if i_episode % 100 == 0 and i_episode != 0:
        print('Episode {}\tAverage Reward: {:.2f}'.format(
            i_episode, np.mean(all_rewards[-100:])))"""

"""# Plot the results
plt.plot(all_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
"""
"""# Save the model
torch.save(model.state_dict(), "Models/DQN_CartPole.pth")"""

# To load the model later
# model.load_state_dict(torch.load("Models/DQN.pth"))
