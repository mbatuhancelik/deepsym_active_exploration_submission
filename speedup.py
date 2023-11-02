from environment import BlocksworldLightning, BlocksWorld_v4
from time import time




env = BlocksworldLightning(gui = 0)
ts1 = time()
for i in range(125):
    action = env.sample_random_action()
    env.step(*action)
ts2 = time()

env = BlocksWorld_v4(gui=0)
ts3 = time()
for i in range(125):
    action = env.sample_random_action()
    env.step(*action)
ts4 = time()

print(f"speedup = {(ts4 - ts3) / (ts2 - ts1)}")