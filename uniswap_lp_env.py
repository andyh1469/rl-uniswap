### Custom Gym Environment for Uniswap v3 Liquidity Provision (ETH-USDC) ###

# Token Convention: Token0 is ETH (x), Token1 is USDC (y)

import logging
import math
import random
import sys

import gym
import numpy as np
from gym import spaces


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
train_logger = logging.getLogger("train_logger")


class UniswapLPEnv(gym.Env):
    def __init__(
        self,
        price_history_df,
        volume_df,
        liquidity_df,
        initial_cash,
        min_tick,
        max_tick,
        tick_spacing,
        token0_decimals,
        token1_decimals,
        fee_weight,
        debug_mode=False,
    ):
        super(UniswapLPEnv, self).__init__()
        # Hourly ETH price history
        self.price_history_df = price_history_df
        # Total liquidity in each tick range (hourly)
        self.liquidity_df = liquidity_df
        # Volume traded in each tick range (hourly)
        self.volume_df = volume_df

        self.TOKEN0_DECIMALS = token0_decimals
        self.TOKEN1_DECIMALS = token1_decimals

        self.NUM_ACTIVE_TICK_RANGES = int((max_tick - min_tick) / tick_spacing)
        self.MIN_TICK = min_tick
        self.MIN_PRICE = self.tick2Price(self.MIN_TICK)
        self.MAX_TICK = max_tick
        self.MAX_PRICE = self.tick2Price(self.MAX_TICK)
        self.TICK_SPACING = tick_spacing
        self.FEE_PCENT = 0.3
        self.INITIAL_CASH = initial_cash
        # We have 24 hours of historical price data (including the current timestep)
        self.ETH_PRICE_HISTORY_OFFSET = 23
        self.FEE_WEIGHT = fee_weight
        self.DEBUG_MODE = debug_mode

        self.ranges_tick_boundaries = self.getRangeTickBoundaries()
        self.position_token0 = np.zeros(self.NUM_ACTIVE_TICK_RANGES)
        self.position_token1 = np.zeros(self.NUM_ACTIVE_TICK_RANGES)
        self.position_cash = initial_cash
        self.new_position_token0 = np.zeros(self.NUM_ACTIVE_TICK_RANGES)
        self.new_position_token1 = np.zeros(self.NUM_ACTIVE_TICK_RANGES)
        self.new_position_cash = 0
        # [(Hold, Buy, Sell), (Whick Tick), (% to Add/Remove)]
        self.action_space_dim = 3
        # Multiply by 4 for Token0, Token1, Volume, Liquidity, add 1 for cash not in Uniswap, add 24 for ETH price over last 24 hours
        self.observation_space_dim = 4 * (self.NUM_ACTIVE_TICK_RANGES) + 1 + self.ETH_PRICE_HISTORY_OFFSET + 1
        self.fee_history = []
        self.position_history = []
        self.current_step = None
        self.starting_step = None
        self.steps_per_episode_list = []
        self.eth_pos_list = []

        # Action space is a discrete tuple. First entry is (0, 1, 2) --> (HOLD, ADD, REMOVE). 2nd entry is (0, 1, ..., N_TICK_RANGE -1) --> which tick to modify. 3rd entry is (0, 1, ..., 100) --> % liquidity to add/remove.
        self.action_space = spaces.MultiDiscrete([3, self.NUM_ACTIVE_TICK_RANGES, 101])
        self.observation_space = spaces.Box(
            np.concatenate(
                [
                    np.zeros(2 * (self.NUM_ACTIVE_TICK_RANGES) + 1),
                    self.MIN_PRICE * np.ones(self.ETH_PRICE_HISTORY_OFFSET + 1),
                    np.zeros(2 * self.NUM_ACTIVE_TICK_RANGES),
                ]
            ),
            np.concatenate(
                [
                    sys.maxsize * np.ones(2 * (self.NUM_ACTIVE_TICK_RANGES) + 1),
                    self.MAX_PRICE * np.ones(self.ETH_PRICE_HISTORY_OFFSET + 1),
                    sys.maxsize * np.ones(self.NUM_ACTIVE_TICK_RANGES),
                    np.ones(self.NUM_ACTIVE_TICK_RANGES),
                ]
            ),
        )

    # Converts a Uniswap3 tick into a human-readable price.
    def tick2Price(self, tick):
        rawprice = 1.0001**tick
        return (10 ** (self.TOKEN0_DECIMALS - self.TOKEN1_DECIMALS)) / rawprice

    def price2Tick(self, price):
        return round(math.log(10 ** (self.TOKEN0_DECIMALS - self.TOKEN1_DECIMALS) / price, 1.0001))

    def getValidTickBounds(self, tick):
        rem = tick % self.TICK_SPACING
        return tick - rem, tick + self.TICK_SPACING - rem

    def getRangeTickBoundaries(self):
        tick_boundaries = []
        cur_lb_tick = self.MIN_TICK

        for i in range(self.NUM_ACTIVE_TICK_RANGES):
            cur_ub_tick = cur_lb_tick + self.TICK_SPACING
            cur_tick_boundary = (cur_lb_tick, cur_ub_tick)
            tick_boundaries.append(cur_tick_boundary)
            cur_lb_tick = cur_ub_tick

        return tick_boundaries

    # Takes the current position (which gives the number of real tokens per liquidity range), and determines how much liquidity is being provided in each range
    # This depends on the current price in relation to the range price boundaries.
    def computeLiquidityProvisionInRanges(self):
        liquidities_provided = np.zeros(self.NUM_ACTIVE_TICK_RANGES)
        for idx, boundary_tuple in enumerate(self.ranges_tick_boundaries):
            amount0 = self.new_position_token0[idx]
            amount1 = self.new_position_token1[idx]

            tick_lower = boundary_tuple[0]
            tick_upper = boundary_tuple[1]

            p_a = self.tick2Price(tick_upper)
            p_b = self.tick2Price(tick_lower)

            cur_price = self.price_history[-1]

            if cur_price <= p_a:
                amount = amount0 * (np.sqrt(p_b) * np.sqrt(p_a)) / (np.sqrt(p_b) - np.sqrt(p_a))
            elif cur_price >= p_b:
                amount = amount1 / (np.sqrt(p_b) - np.sqrt(p_a))
            else:
                amount = amount1 / (cur_price - np.sqrt(p_a))

            liquidities_provided[idx] = amount

        return liquidities_provided

    def updatePositions(self, cur_price):
        mixed_tick = self.price2Tick(cur_price)
        mixed_tick_lower, mixed_tick_upper = self.getValidTickBounds(mixed_tick)
        mixed_tick_lower_idx = int((mixed_tick_lower - self.MIN_TICK) / self.TICK_SPACING)

        # Convert all Token1 (USDC) into Token0 (ETH) for all ticks above the mixed tick
        for i in range(mixed_tick_lower_idx):
            if self.liquidities_provided[i] == 0:
                continue

            p_b = self.tick2Price(i * self.TICK_SPACING + self.MIN_TICK)
            p_a = self.tick2Price((i + 1) * self.TICK_SPACING + self.MIN_TICK)

            self.new_position_token0[i] = self.liquidities_provided[i] * (1 / np.sqrt(p_a) - 1 / np.sqrt(p_b))
            self.new_position_token1[i] = 0

        # Convert all Token0 (ETH) into Token1 (USDC) for all ticks below the mixed tick
        for i in range(mixed_tick_lower_idx + 1, self.NUM_ACTIVE_TICK_RANGES):
            if self.liquidities_provided[i] == 0:
                continue

            p_b = self.tick2Price(i * self.TICK_SPACING + self.MIN_TICK)
            p_a = self.tick2Price((i + 1) * self.TICK_SPACING + self.MIN_TICK)

            self.new_position_token1[i] = self.liquidities_provided[i] * (np.sqrt(p_b) - np.sqrt(p_a))
            self.new_position_token0[i] = 0

        # Convert mixed tick to have the proper ratio of Token0 and Token1
        p_b = self.tick2Price(mixed_tick_lower_idx * self.TICK_SPACING + self.MIN_TICK)
        p_a = self.tick2Price((mixed_tick_lower_idx + 1) * self.TICK_SPACING + self.MIN_TICK)
        self.new_position_token0[mixed_tick_lower_idx] = (
            self.liquidities_provided[mixed_tick_lower_idx] * (np.sqrt(p_b) - np.sqrt(cur_price)) / (np.sqrt(cur_price) * np.sqrt(p_b))
        )
        self.new_position_token1[mixed_tick_lower_idx] = self.liquidities_provided[mixed_tick_lower_idx] * (np.sqrt(cur_price) - np.sqrt(p_a))

    # Reset the state of the environment to an initial state
    def reset(self):
        ## No LP Positions, only initial cash
        self.position_token0 = np.zeros(self.NUM_ACTIVE_TICK_RANGES)
        self.position_token1 = np.zeros(self.NUM_ACTIVE_TICK_RANGES)
        self.new_position_token0 = np.zeros(self.NUM_ACTIVE_TICK_RANGES)
        self.new_position_token1 = np.zeros(self.NUM_ACTIVE_TICK_RANGES)
        self.position_cash = self.INITIAL_CASH

        # Set the current step to a random point within the data frame (but need to make sure it's between step 0 and the very last step, i.e. don't consider negative index ETH price indices)
        self.current_step = random.randint(0, len(self.price_history_df.index) - 2)
        self.starting_step = self.current_step
        total_position = np.concatenate(
            [
                self.position_token1,
                self.position_token0,
                np.array(self.position_cash).reshape(
                    1,
                ),
            ]
        )

        self.price_history = np.array(self.price_history_df.loc[self.current_step][2:])
        cur_price = self.price_history[-1]
        self.starting_price = cur_price
        self.volume_ticks = np.array(self.volume_df.loc[self.current_step][2:])
        total_liquidity_ticks = np.array(self.liquidity_df.loc[self.current_step + self.ETH_PRICE_HISTORY_OFFSET][1:])
        self.liquidities_provided = self.computeLiquidityProvisionInRanges()
        self.fraction_liquidity_ticks = self.liquidities_provided / total_liquidity_ticks
        self.fee_history = []

        obs = np.concatenate([total_position, self.price_history, self.volume_ticks, self.fraction_liquidity_ticks])

        return obs

    def takeAction(self, action):
        trade_type = action[0]
        tick_idx = action[1]

        tick = tick_idx * self.TICK_SPACING + self.MIN_TICK
        pcent_change = action[2] / 100

        cur_price = self.price_history[-1]
        mixed_tick = self.price2Tick(cur_price)
        # By convention, the lower tick bound is converted to the index used to find the tick in the observation
        mixed_tick_lower, mixed_tick_upper = self.getValidTickBounds(mixed_tick)
        # mixed_tick_idx is not a integer, but a float number
        mixed_tick_lower_idx = int((mixed_tick_lower - self.MIN_TICK) / self.TICK_SPACING)

        # HOLD CASE
        if trade_type == 0:
            self.new_position_token0 = self.position_token0.copy()
            self.new_position_token1 = self.position_token1.copy()
            self.new_position_cash = self.position_cash
        # ADD CASE (in this case, the % change is of the CASH)
        elif trade_type == 1:
            # Mixed Case
            if tick_idx == mixed_tick_lower_idx:
                total_value_added = pcent_change * self.position_cash  # (k)
                p_a = self.tick2Price(mixed_tick_upper)
                p_b = self.tick2Price(mixed_tick_lower)

                # Obtained by solving system of eqs
                price_ratio_term = (np.sqrt(cur_price) * np.sqrt(p_b) * (np.sqrt(cur_price) - np.sqrt(p_a))) / (np.sqrt(p_b) - np.sqrt(cur_price))
                token0_delta = total_value_added / (cur_price + price_ratio_term)  # (x)
                token1_delta = total_value_added - token0_delta * cur_price  # (y)

                self.new_position_token0[tick_idx] = self.position_token0[tick_idx] + token0_delta
                self.new_position_token1[tick_idx] = self.position_token1[tick_idx] + token1_delta
                self.new_position_cash = self.position_cash - total_value_added

            # Upper bound of the liquidity interval is less than the current price,
            # Thus LP needs to invest USDC into the pool, i.e., Token0
            elif tick_idx > mixed_tick_lower_idx:
                self.new_position_token1[tick_idx] = self.position_token1[tick_idx] + (pcent_change * self.position_cash)
                self.new_position_token0[tick_idx] = self.position_token0[tick_idx].copy()
                self.new_position_cash = self.position_cash - pcent_change * self.position_cash
            # Token1 Only --> put ETH into the pool
            else:
                self.new_position_token0[tick_idx] = self.position_token0[tick_idx] + (pcent_change * self.position_cash) / cur_price
                self.new_position_token1[tick_idx] = self.position_token1[tick_idx].copy()
                self.new_position_cash = self.position_cash - pcent_change * self.position_cash

        # REMOVE CASE (In this case, the % change is of the TICK)
        else:
            # Mixed Case
            if tick_idx == mixed_tick_lower_idx:
                total_value_removed = pcent_change * (cur_price * self.position_token0[tick_idx] + self.position_token1[tick_idx])  # (k)
                p_a = self.tick2Price(mixed_tick_upper)
                p_b = self.tick2Price(mixed_tick_lower)

                ## Obtained by solving system of eqns
                price_ratio_term = (np.sqrt(cur_price) * np.sqrt(p_b) * (np.sqrt(cur_price) - np.sqrt(p_a))) / (np.sqrt(p_b) - np.sqrt(cur_price))
                token0_delta = total_value_removed / (cur_price + price_ratio_term)  # (x)
                token1_delta = total_value_removed - token0_delta * cur_price  # (y)

                self.new_position_token0[tick_idx] = self.position_token0[tick_idx] - token0_delta
                self.new_position_token1[tick_idx] = self.position_token1[tick_idx] - token1_delta
                self.new_position_cash = self.position_cash + total_value_removed

            # Upper bound of the liquidity interval is less than the current price,
            # Thus LP needs to invest USDC into the pool, i.e., Token0
            elif tick_idx > mixed_tick_lower_idx:
                self.new_position_token1[tick_idx] = self.position_token1[tick_idx] - pcent_change * self.position_token1[tick_idx]
                self.new_position_token0[tick_idx] = self.position_token0[tick_idx].copy()
                self.new_position_cash = self.position_cash + pcent_change * self.position_token1[tick_idx]

            ## Token1 Only --> put ETH into the pool
            else:
                self.new_position_token0[tick_idx] = self.position_token0[tick_idx] - pcent_change * self.position_token0[tick_idx]
                self.new_position_token1[tick_idx] = self.position_token1[tick_idx].copy()
                self.new_position_cash = self.position_cash + cur_price * pcent_change * self.position_token0[tick_idx]

        if self.DEBUG_MODE:
            train_logger.info(f"Current ETH Price: {cur_price}")
            train_logger.info(f"Current Position Token0: {self.position_token0}")
            train_logger.info(f"Current Position Token1: {self.position_token1}")
            train_logger.info(f"Current Position Cash: {self.position_cash}")
            train_logger.info(f"New Position Token0: {self.new_position_token0}")
            train_logger.info(f"New Position Token1: {self.new_position_token1}")
            train_logger.info(f"New Position Cash: {self.new_position_cash}")

        return

    def computeReward(self):
        pre_price = self.price_history[-1]

        # Retrieve next hour's data
        self.price_history = np.array(self.price_history_df.loc[self.current_step][2:])
        cur_price = self.price_history[-1]
        self.volume_ticks = np.array(self.volume_df.loc[self.current_step][2:])
        total_liquidity_ticks = np.array(self.liquidity_df.loc[self.current_step + self.ETH_PRICE_HISTORY_OFFSET][1:])
        self.liquidities_provided = self.computeLiquidityProvisionInRanges()
        self.fraction_liquidity_ticks = self.liquidities_provided / total_liquidity_ticks

        self.updatePositions(cur_price)

        cur_fee = np.sum(self.FEE_PCENT / 100 * self.volume_ticks * self.fraction_liquidity_ticks)
        self.fee_history.append(cur_fee)

        if self.current_step >= len(self.price_history_df.index) - 2:
            reward = (
                cur_price * np.sum(self.new_position_token0)
                + np.sum(self.new_position_token1)
                + self.new_position_cash
                + self.FEE_WEIGHT * sum(self.fee_history)
            ) - self.INITIAL_CASH
        else:
            reward = 0

        return reward

    def nextObservation(self):
        self.position_token0 = self.new_position_token0.copy()
        self.position_token1 = self.new_position_token1.copy()
        self.position_cash = self.new_position_cash

        total_position = np.concatenate(
            [
                self.position_token1,
                self.position_token0,
                np.array(self.position_cash).reshape(
                    1,
                ),
            ]
        )

        obs = np.concatenate([total_position, self.price_history, self.volume_ticks, self.fraction_liquidity_ticks])

        return obs

    def step(self, action):
        # Check if current_step exceeds the csv length
        done = self.current_step >= len(self.price_history_df.index) - 2

        # Execute one time step within the environment
        self.takeAction(action)
        self.current_step += 1
        reward = self.computeReward()
        obs = self.nextObservation()

        if done:
            eth_end_price = np.array(self.price_history_df.loc[self.current_step][2:])[-1]
            eth_strategy_position = (self.INITIAL_CASH / self.starting_price) * eth_end_price
            self.steps_per_episode_list.append(self.current_step - self.starting_step)
            self.eth_pos_list.append(eth_strategy_position)

        return obs, reward, done, {}
