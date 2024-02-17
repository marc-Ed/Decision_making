import gym
from stable_baselines3 import A2C, PPO
import os

env = gym.make("LunarLander-v2", render_mode="human")
# env = gym.make("LunarLander-v2")
env.reset()

models_dir = "Models"
model_path = f"{models_dir}/110000.zip"

model = PPO.load(model_path, env=env)

episodes = 10
for ep in range(episodes):
    obs, _ = env.reset()
    done = False

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)


env.close()
