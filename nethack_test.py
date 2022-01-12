import gym
import nle
import time
import matplotlib.pyplot as plt

env = gym.make("NetHackScore-v0")
obs = env.reset()
env.render()
for key in obs:
    print(key, obs[key].shape)
print("-"*80)
for r in obs["chars"]:
    for c in r:
        print(chr(c), end="")
    print()
print("-"*80)
for r in obs["colors"]:
    for c in r:
        print(chr(c), end="")
    print()
print("-"*80)
print(env.action_space)
print(env.observation_space)
