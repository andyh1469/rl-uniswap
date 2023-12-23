import os
import time

import yaml
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from uniswap_lp_env import UniswapLPEnv, train_logger

if __name__ == "__main__":
    with open("config.yml", "r") as yml_file:
        config_data = yaml.safe_load(yml_file)

    env_vars = config_data["env_vars"]
    train_params = config_data["train_params"]

    train_logger.info("Environment Variables")
    for k, v in env_vars.items():
        train_logger.info(f"{k}: {v}")
    train_logger.info("---------------------------------------------------")

    train_logger.info("Training Parameters")
    for k, v in train_params.items():
        train_logger.info(f"{k}: {v}")
    train_logger.info("---------------------------------------------------")

    # env variables
    INITIAL_CASH = env_vars["INITIAL_CASH"]
    MIN_TICK = env_vars["MIN_TICK"]
    MAX_TICK = env_vars["MAX_TICK"]
    TICK_SPACING = env_vars["TICK_SPACING"]
    TOKEN0_DECIMALS = env_vars["TOKEN0_DECIMALS"]
    TOKEN1_DECIMALS = env_vars["TOKEN1_DECIMALS"]
    FEE_WEIGHT = env_vars["FEE_WEIGHT"]
    DEBUG_MODE = env_vars["DEBUG_MODE"]

    # training params
    n_nodes = train_params["n_nodes"]
    lr = train_params["lr"]
    n_steps = train_params["n_steps"]
    num_episodes = train_params["num_episodes"]

    # load data
    price_history_df = pd.read_csv("blockchain_data/price_history_array_050521_102922.csv")
    volume_df = pd.read_csv("blockchain_data/volume_array_050521_102922.csv", encoding="latin-1")
    liquidity_df = pd.read_csv("blockchain_data/total_liquidity_array_050521_102922.csv")

    # policy training
    env = DummyVecEnv(
        [
            lambda: UniswapLPEnv(
                price_history_df,
                volume_df,
                liquidity_df,
                INITIAL_CASH,
                MIN_TICK,
                MAX_TICK,
                TICK_SPACING,
                TOKEN0_DECIMALS,
                TOKEN1_DECIMALS,
                FEE_WEIGHT,
                DEBUG_MODE,
            )
        ]
    )
    policy_kwargs = {"net_arch": {"pi": [n_nodes, n_nodes], "vf": [n_nodes, n_nodes]}}
    model = PPO("MlpPolicy", env, learning_rate=lr, n_steps=n_steps, policy_kwargs=policy_kwargs, verbose=0)

    start = time.time()
    model.learn(total_timesteps=round(len(price_history_df), -3) * num_episodes)
    end = time.time()

    train_logger.info(f"Training finished in {end - start} sec")
    train_logger.info(f"Collected {len(env.get_attr('eth_pos_list')[0])} episodes of data")

    if not os.path.exists("results"):
        os.makedirs("results")

    # Plot Results
    g = plt.gca()
    plt.plot(env.get_attr("steps_per_episode_list")[0], "-o")
    plt.axhline(y=INITIAL_CASH, color="black")
    plt.plot(env.get_attr("eth_pos_list")[0], "-o")
    plt.title("Uniswap Agent Policy Training Results")
    plt.ylabel("Position Value")
    plt.xlabel("Episode")
    plt.legend(["Agent Position Value", "Starting Position", "100% ETH Position Value"])
    g.axes.get_xaxis().set_visible(False)
    plt.savefig(f"results/rl_train {time.asctime()}.png")
