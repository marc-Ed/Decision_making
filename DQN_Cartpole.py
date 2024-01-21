import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


# Define the neural network model


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.out(x)
        return x


# Initialize environment and model
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")

model = DQN()
LR = 1e-6
optimizer = optim.Adam(model.parameters(), lr=LR, amsgrad=True)
loss_fn = nn.CrossEntropyLoss()

gamma = 0.9
# Function to calculate discounted rewards


def discount_reward(rewards, gamma):
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r


# Training loop
n_episodes = 1500
max_time = 200
all_rewards = []
train_data = deque(maxlen=10000)  # Use deque to keep the last N experiences

for i_episode in range(n_episodes):
    obs, _ = env.reset()
    # Cart position / Cart velocity / pole angle / pole angular velocity
    state = torch.from_numpy(obs)
    episode_reward = 0
    states_hist = []
    actions_hist = []
    rewards_hist = []

    for t in range(max_time):
        # Forward pass to get action probabilities
        q_values = model(state)
        action_probs = F.softmax(
            q_values, dim=0).detach().numpy()
        # print(action_probs)

        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        # action = t % 2
        # print(action)

        # Take a step in the environment
        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        # print(next_state, reward)

        # print("state, next state", state.shape, next_state.shape)

        # Store the transition
        states_hist.append(state.numpy())
        actions_hist.append(action)
        rewards_hist.append(reward)

        state = next_state
        episode_reward += reward

        if done:
            all_rewards.append(episode_reward)
            discounted_rewards = discount_reward(rewards_hist, gamma)
            train_data.extend(
                zip(states_hist, actions_hist, discounted_rewards))

            # Every 10 episodes, perform a training step
            if i_episode % 10 == 0 and i_episode != 0:
                ep_data = np.array(list(train_data), dtype=object)
                states = np.stack([item[0] for item in ep_data])
                states_tensor = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor([item[1] for item in ep_data]).long()
                rewards = torch.tensor([item[2] for item in ep_data]).float()

                # Calculate loss and backpropagate
                optimizer.zero_grad()
                action_probabilities = F.softmax(
                    model(states_tensor), dim=1)
                # print("action_probabilities", action_probabilities)
                # print("actions", actions)
                loss = loss_fn(action_probabilities, actions)
                # print("loss", loss)
                loss.backward()
                optimizer.step()

                # Clear training data
                train_data.clear()
            break

    if i_episode % 100 == 0 and i_episode != 0:
        print('Episode {}\tAverage Reward: {:.2f}'.format(
            i_episode, np.mean(all_rewards[-100:])))

# Plot the results
plt.plot(all_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

# Save the model
torch.save(model.state_dict(), "Models/DQN_CartPole.pth")

# To load the model later
# model.load_state_dict(torch.load("Models/DQN.pth"))
