# LosingMoney
A project to losing money by trading stock

## feature engineering
In `analysis/corr_check.ipynb`, we process raw data and get total 40 features.

Five basic features are:

| close | open | high | low  | volume/K |
| ----- | ---- | ---- | ---- | -------- |

By those basic features, we can get some simple indicators.

| daliy_region | daily_opengian | daily_gain | ln_Rt |
| ------------ | -------------- | ---------- | ----- |

With economical knowledges, we know there are some classic indicators. They are:

| MACD   | CCI      | MIDPRICE    | SAR      |
| ------ | -------- | ----------- | -------- |
| MACD_l | DX       | HT_DCPERIOD | ADX      |
| MAMA,  | MINUS_DI | HT_DCPHASE  | TRIX     |
| MAMA_l | MINUS_DM | HT_PHASOR   | AROONOSC |
| K      | J        | OBV         | RSI      |
| OC     | CMO      |             |          |

We also refer to ***WorldQuant Formulaic 101 Alphas*** to get some alpha indicators by data mining. 

| alpha_12 | alpha_101 | alpha_54 | alpha_23 |
| -------- | --------- | -------- | -------- |

The heatmap is showed as below.

![image-20221228171449388](https://cdn.jsdelivr.net/gh/frinkleko/PicgoPabloBED@master/images_for_wechat/image-20221228171449388.png)

One thing should be done is features selecting. We haven’t done this.

## LSTM for prediction

 In `model` folder we provided prediction solution with LTSM model.

One noticeable thing is that we change the loss function for better  fitting. Generally, MSE is the loss function for time-series forecasting tasks. But in many specific tasks, predicted time-series data often perform amplitude fluctuations and time delay fluctuations. Those fluctuations often means disasters for stock markets which acquire correct timing and right amplitude predictions.  Simple MSE may hard to guide model to notice this. 

DTW, Soft-DTW, DILATE are loss functions focus on these below mentioned fluctuations, which can better represent the “**diff**” of two time-series. We choose this as a final index. During training, we realize “**percentage**” is the thing we care in stock price predictions. However, simply MAPE can’t help to converge quickly and correctly. Finally we use the weighted sum of MSE and MAPE. The weights is decided by eval result on both train and valuate set. We hope the model is leaded by MSE firstly, and gradually care more about MAPE.

Loss: MSE + 0.3 * MAPE

We have tried several schedulers. We are encouraged by **fast.ai**‘s concpet , we firstly try **OneCycleLR** , and result proved it is not the best :cry: . 

We finally get MSE of [0.003-0.005],MAPE of [0.01-0.014]

```bash
The mean squared error of stock ['close', 'open', 'high', 'low'] is  [0.00572115 0.00338099 0.00487787 0.00325543]
The mean squared percentage error of stock close is 0.01433726164531638
The mean squared percentage error of stock open is 0.010846512384708686
The mean squared percentage error of stock high is 0.013249862787169227
The mean squared percentage error of stock low is 0.010634444940022284
```

## Trading 

We use xxxx as baseline. One parameter for this strategy is `setwater`, which represents the aggressive level. 

The strategy can be described as below:





However, considering our model certainly have some errors, we also designed several techniques to acquire best return.

### BIAS exists!

As we just mentioned, our strategy feels terrible when predicted high is actually lower than low truth (same as the predicted low). So the set a bias to “adjust” the predicted high and low to have more chances to trade is a natural idea.

The bias was calculated as max(MAPE)*mean(val of train data), which is finally 50. Therefore, the` high[i] `and` low[i]`change to ` high[i]+50 `and` low[i]-50`.

The return curve is labled 

### I’m OUT

Model do have errors. And it may become continual mistakes. We also want to avoid “unpredictable” period. Therefore, we use rolling  windows to calculate last 7 days return. If 

