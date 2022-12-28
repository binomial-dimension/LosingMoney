# LosingMoney
A project to losing money by trading stock

## feature engineering
In `analysis/corr_check.ipynb`, we process raw data and get total 40 features.

Five basic features are:

+ close
+ open
+ high
+ low
+ volume/K

By those basic features, we can get some simple indicators.

+ daliy_region
+ daily_opengian
+ daily_gain
+ ln_Rt

With economical knowledges, we know there are some classic indicators. They are:

| MACD   | CCI      | MIDPRICE    | SAR      |
| ------ | -------- | ----------- | -------- |
| MACD_l | DX       | HT_DCPERIOD | ADX      |
| MAMA,  | MINUS_DI | HT_DCPHASE  | TRIX     |
| MAMA_l | MINUS_DM | HT_PHASOR   | AROONOSC |
| K      | J        | OBV         | RSI      |
| OC     | CMO      |             |          |

We also refer to ***WorldQuant Formulaic 101 Alphas*** to get some alpha indicators by data mining. 

+ alpha_12
+ alpha_101
+ alpha_54
+ alpha_23

The heatmap is showed as below.

![image-20221228171449388](https://cdn.jsdelivr.net/gh/frinkleko/PicgoPabloBED@master/images_for_wechat/image-20221228171449388.png)

One thing should be done is features selecting. We haven’t done this.

## LSTM for prediction

 In `model` folder we provided prediction solution with both LSTM and transformer, we finally use LTSM.

One noticeable thing is that we change the loss function for better  fitting. Generally, MSE is the loss function for time-series forecasting tasks. But in many specific tasks, predicted time-series data often perform amplitude fluctuations and time delay fluctuations. Those fluctuations often means disasters for stock markets which acquire correct timing and right amplitude predictions.  Simple MSE may hard to guide model to notice this. 

DTW, Soft-DTW, DILATE are loss functions focus on these below mentioned fluctuations, which can better represent the “**diff**” of two time-series. We choose this as a final index. During training, we realize “**percentage**” is the thing we care in stock price predictions. However, simply MAPE can’t help to converge quickly and correctly. Finally we use the weighted sum of MSE and MAPE. The weights is decided by eval result on both train and valuate set. We hope the model is leaded by MSE firstly, and gradually care more about MAPE.

Results of test set prove it. We finally get MSE of [0.003-0.008],MAPE of [0.01-0.016]

```
The mean squared error of stock ['close', 'open', 'high', 'low'] is [0.00796766 0.00338913 0.00446633 0.00744549]
The mean squared percentage error of stock close is 0.016048249795682618
The mean squared percentage error of stock open is 0.010534894554325255
The mean squared percentage error of stock high is 0.011733987383671367
The mean squared percentage error of stock low is 0.015706365304477363
```

## Trading 

We use xxxx as baseline. One parameter for this strategy is `setwater`, which represents the aggressive level. See more results in following.

![image-20221228183732521](https://cdn.jsdelivr.net/gh/frinkleko/PicgoPabloBED@master/images_for_wechat/image-20221228183732521.png)

![image-20221228183741095](https://cdn.jsdelivr.net/gh/frinkleko/PicgoPabloBED@master/images_for_wechat/image-20221228183741095.png)