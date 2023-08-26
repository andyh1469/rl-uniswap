import pandas as pd
import time
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from uniswap_lp_env import UniswapLPEnv

# Env Variables
INITIAL_CASH = 10000
MIN_TICK = 191340
MAX_TICK = 208500 + 60
TICK_SPACING = 60
TOKEN0DECIMALS = 18
TOKEN1DECIMALS = 6 
NUM_ACTIVE_TICK_RANGES = int((MAX_TICK - MIN_TICK)/TICK_SPACING)
ETH_PRICE_HISTORY_OFFSET = 23

# Training Params
n_nodes = 64
lr = 0.03
n_steps = 256
feeWeight = 1000
num_episodes = 5

# Load Data
price_history_df = pd.read_csv('blockchain_data/price_history_array_050521_102922.csv')
volume_df = pd.read_csv('blockchain_data/volume_array_050521_102922.csv', encoding='latin-1')
liquidity_df = pd.read_csv('blockchain_data/total_liquidity_array_050521_102922.csv')

# Policy Training
env = DummyVecEnv([lambda: UniswapLPEnv(price_history_df, volume_df, liquidity_df, INITIAL_CASH, MIN_TICK, MAX_TICK, TICK_SPACING, TOKEN0DECIMALS, TOKEN1DECIMALS, feeWeight)])
policy_kwargs = dict(net_arch={'pi':[n_nodes, n_nodes], 'vf':[n_nodes, n_nodes]})
model = PPO('MlpPolicy', env, learning_rate=lr, n_steps = n_steps, policy_kwargs=policy_kwargs, verbose=0)

start = time.time()
model.learn(total_timesteps=13000*num_episodes)
end = time.time()

print('Learning took', end - start, 'sec')
print('Collected {} episodes of data'.format(len(env.get_attr('eth_pos_list')[0])))

# Plot Results
g = plt.gca()
plt.plot(env.get_attr('steps_per_episode_list')[0], '-o')
plt.axhline(y=INITIAL_CASH, color='black')
plt.plot(env.get_attr('eth_pos_list')[0], '-o')
plt.title('Uniswap Agent Policy Training Results')
plt.ylabel('Position Value')
plt.xlabel('Episode')
plt.legend(['Agent Position Value', 'Starting Position', '100% ETH Position Value'])
g.axes.get_xaxis().set_visible(False)
plt.show()