import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

from wy_baseline import wy_tradegy_int

np.random.seed(0)

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 6000
MAX_VOLUME = 1000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000
MAX_DAY_CHANGE = 1

INITIAL_ACCOUNT_BALANCE = 100000


"""
A stock trading environment for OpenAI gym
"""
class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4*41 + 6,), dtype=np.float16)
        self.percent_token = 0.1


    def _next_observation(self):
        obs1 = np.array([
            self.df.loc[self.current_step - 40:self.current_step, 'open_truth'] +
            self.df.loc[self.current_step, 'open'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - 40:self.current_step, 'high_truth'] +
            self.df.loc[self.current_step, 'high'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - 40:self.current_step, 'low_truth'] +
            self.df.loc[self.current_step, 'low'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - 40:self.current_step, 'close_truth'] +
            self.df.loc[self.current_step, 'close'] / MAX_SHARE_PRICE,
        ])
        obs1 = obs1.flatten()
        obs2 = np.array([
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ])
        obs = np.concatenate((obs1, obs2), axis=0)
        return obs


    # Set the current price to a random price within the time step
    def _take_action(self, action):
        current_price = random.uniform(
            self.df.loc[self.current_step, "open_truth"], self.df.loc[self.current_step, "close_truth"])

        action_type = action[0]
        amount = action[1]
        # if action_type < 1:
        #     # Buy amount % of balance in shares
        #     total_possible = int(self.balance / current_price)
        #     shares_bought = int(total_possible * amount)
        #     prev_cost = self.cost_basis * self.shares_held
        #     additional_cost = shares_bought * current_price

        #     fee = shares_bought/10000*1.1
        #     additional_cost += fee

        #     self.balance -= additional_cost
        #     self.cost_basis = (
        #         prev_cost + additional_cost) / (self.shares_held + shares_bought)
        #     self.shares_held += shares_bought

        # elif action_type < 2:
        #     # Sell amount % of shares held
        #     shares_sold = int(self.shares_held * amount)

        #     fee = shares_sold/10000*1.1
        #     shares_sold -= fee

        #     self.balance += shares_sold * current_price
        #     self.shares_held -= shares_sold
        #     self.total_shares_sold += shares_sold
        #     self.total_sales_value += shares_sold * current_price

        #     self.net_worth = self.balance + self.shares_held * current_price
        if action_type < 1:
            self.percent_token += amount
            self.percent_token = min(self.percent_token, 1)
            self.percent_token = max(self.percent_token, 0)
        elif action_type < 2:
            self.percent_token -= amount
            self.percent_token = min(self.percent_token, 1)
            self.percent_token = max(self.percent_token, 0)
        predhigh = self.df.loc[self.current_step, 'high']
        predlow = self.df.loc[self.current_step, 'low']
        truehigh = self.df.loc[self.current_step, 'high_truth']
        truelow = self.df.loc[self.current_step, 'low_truth']
        trueopen = self.df.loc[self.current_step, 'open_truth']
        trueclose = self.df.loc[self.current_step, 'close_truth']
        if action_type < 2:
            self.balance *= wy_tradegy_int(predhigh, predlow, truehigh, truelow, trueopen,
                                        trueclose, setwater=self.percent_token, maxm=self.balance*1000)
            self.balance *= 0.99989
        self.net_worth = self.balance
        self.shares_held = int(
            self.balance / current_price) * self.percent_token
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0


    # Execute one time step within the environment
    def step(self, action):
        self._take_action(action)
        done = False

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'open_truth'].values) - 1:
            self.current_step = 41  # loop training

        # profits
        reward = self.net_worth / \
            INITIAL_ACCOUNT_BALANCE - 1  # 1.0

        if self.net_worth <= 0:
            done = True

        obs = self._next_observation()

        return obs, reward, done, {}


    # Reset the state of the environment to an initial state
    def reset(self, new_df=None):
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.percent_token = 0.1
        # pass test dataset to environment
        if new_df:
            self.df = new_df

        # set current step to 0
        self.current_step = 41

        return self._next_observation()


    # Render the environment to the screen
    def render(self, mode='human', close=False):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        return self.balance, self.net_worth, profit
