import gym
from stable_baselines3 import A2C, PPO
import os
import tensorflow as tf
print(tf.__version__)

models_dir = "Models"
logdir = 'Logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# env = gym.make("LunarLander-v2", render_mode="human")
env = gym.make("LunarLander-v2")

env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
TIMESTEPS = 10000  # Intervals to show results
INTERVALS = 30
for i in range(1, INTERVALS):
    model.learn(total_timesteps=TIMESTEPS,
                reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

"""print("sample action :", env.action_space.sample())
print("obs space shape", env.observation_space.shape)
print("sample obs space", env.observation_space.sample())
episodes = 10
for ep in range(episodes):
    obs = env.reset()
    done = False

    while not done:
        env.render()
        obs, reward, done, _, _ = env.step(env.action_space.sample())"""


env.close()
