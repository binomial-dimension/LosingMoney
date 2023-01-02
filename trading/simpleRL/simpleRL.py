import os
import pickle
import pandas as pd
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import logging

from env import StockTradingEnv

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

np.random.seed(0)

def stock_trade(stock_file):
    # read train set
    df = pd.read_csv(stock_file)

    # create a environment
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    # check if model exist
    if os.path.exists('./model/stock_trade.zip'):
        logging.info('load model')
        model = PPO2.load('./model/stock_trade', env=env)
    else:
        logging.info('new model')
        model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./log')

    # train model
    model.learn(total_timesteps=int(1e4), log_interval=1e3)
    logging.info('save model')
    # save model
    model.save('./model/stock_trade')

    # read test set
    df_test = pd.read_csv('../../data/predict.csv')

    # create a environment
    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    obs = env.reset()

    # record all states during the test
    actions = []
    blances = []
    net_worths = []
    profits = []

    for i in range(len(df_test) - 1):
        # for each step, predict the action from observation
        action, _states = model.predict(obs)

        # take the action and get the next observation, reward, done
        obs, rewards, done, info = env.step(action)
        # record the state
        actions.append(action)

        # get the current state
        blance, net_worth, profit = env.render()
        blances.append(blance)
        net_worths.append(net_worth)
        profits.append(profit)

        if done:
            break
    return actions, np.array(blances), np.array(net_worths), np.array(profits)


def test_a_stock_trade(stock_code='hs300'):
    # train set path
    stock_file = '../../data/predict_train.csv'

    # train and test
    actions, blances, net_worths, profits = stock_trade(stock_file)

    # plot the result
    fig, ax = plt.subplots(figsize=(20, 10))

    # make the buy and sell point
    buy = []
    for i in range(len(actions)):
        if actions[i][0][0] < 1:
            buy.append(i)
    sell = []
    for i in range(len(actions)):
        if actions[i][0][0] > 1 and actions[i][0][0] < 2:
            sell.append(i)

    # read the truth
    df_test = pd.read_csv('../../data/predict.csv')
    begin = df_test.close_truth[0]
    close_truth = np.array(df_test.close_truth)
    close = np.array(df_test.close)

    # plot the truth and buy and sell point
    ax.plot(close_truth/begin, label='truth')
    ax.plot(close_truth/begin, marker='^',
            label='buy', markevery=buy, markersize=5)
    ax.plot(close_truth/begin, marker='v',
            label='sell', markevery=sell, markersize=5)

    # plot the net_worth
    #ax.plot(blances/blances[0], label='blance')
    ax.plot(net_worths/net_worths[0], label='net_worth')
    #ax.plot(profits/100, label='profit (x100)')

    plt.xlabel('Time [days]')
    ax.legend()
    plt.savefig(f'./img/{stock_code}.png')


if __name__ == '__main__':

    test_a_stock_trade()
