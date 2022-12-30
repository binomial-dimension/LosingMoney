import os
import pickle
import pandas as pd
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from env import StockTradingEnv

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

np.random.seed(0)

def stock_trade(stock_file):

    df = pd.read_csv(stock_file)

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    if os.path.exists('./model/stock_trade'):
        model = PPO2.load('./model/stock_trade',env=env)
    else:
        model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./log')
    
    model.learn(total_timesteps=int(1e7),log_interval=1e3)
    try:
        os.remove('./model/stock_trade')
        model.save('./model/stock_trade')
    except:
        print('model already exists')

    df_test = pd.read_csv('../../data/predict.csv')

    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    obs = env.reset()

    # record all states during the test
    actions = []
    blances = []
    net_worths = []
    profits = []

    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        actions.append(action)

        blance,net_worth,profit = env.render()

        blances.append(blance)
        net_worths.append(net_worth)
        profits.append(profit)

        if done:
            break
    return actions,np.array(blances),np.array(net_worths),np.array(profits)

def test_a_stock_trade(stock_code='hs300'):
    stock_file = '../../data/predict_train.csv'

    actions,blances,net_worths,profits = stock_trade(stock_file)
    #output = pd.DataFrame({'actions':actions.flatten(),'blances':blances.flatten(),'net_worths':net_worths.flatten(),'profits':profits.flatten()})
    #output.to_csv(f'./output/{stock_code}.csv',index=False)

    fig, ax = plt.subplots(figsize = (20,10))

    buy = []
    for i in range(len(actions)):
        if actions[i][0][0] <1:
            buy.append(i)
    sell = []
    for i in range(len(actions)):
        if actions[i][0][0] >1 and actions[i][0][0] <2:
            sell.append(i)

    df_test = pd.read_csv('../../data/predict.csv')
    begin = df_test.close_truth[0]
    close_truth = np.array(df_test.close_truth)
    close = np.array(df_test.close)

    ax.plot(close_truth/begin, label='truth')
    ax.plot(close_truth/begin,marker='^',label='buy',markevery=buy,markersize=5)
    ax.plot(close_truth/begin,marker='v',label='sell',markevery=sell,markersize=5)

    #ax.plot(blances/blances[0], label='blance')
    ax.plot(net_worths/net_worths[0], label='net_worth')
    #ax.plot(profits/100, label='profit (x100)')

    plt.xlabel('Time [days]')

    ax.legend()
    plt.savefig(f'./img/{stock_code}.png')

if __name__ == '__main__':

    test_a_stock_trade()
