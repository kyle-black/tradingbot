
import tensorflow as tf
from tensorflow import keras

#print(tf.version.VERSION)


import os

os.environ['TF_CCP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd

import tensorflow as tf
#from cryptowatch1 import BTCUSD
import cryptowatch1
from sklearn.preprocessing import StandardScaler
#from model_train import import_df
from datetime import datetime
from statistics import mean


#BTCUSD =cryptowatch1.get_ohlc('BTCUSD','30m')
def import_df(df_):

    df_['change'] = df_['close'].pct_change(periods =24)
    df_['change_price'] = df_['close'].shift(24) + (df_['change'] * df_['close'].shift(24))
    df_= df_.dropna()
   # df_ = df_[-48:]
    df =df_[:-24]
    predict_df =df_[-24:]

    #predict_time = predict_df['time']
    scaler = StandardScaler()


    #plot_cols =['time', 'open','high','low','close','vol_','volume', 'RSI', 'MACD','macsignal', 'macdhist', 'ADX', 
    #'Aroon', 'Trendmode', 'AD', 'ADOSC', 'OBV', 'NATR', 'TRANGE'] 
    plot_cols = ['time', 'change','change_price','open', 'high', 'low', 'close', 'vol_', 'volume', 'RSI', 'MACD',
       'macsignal', 'macdhist', 'ADX', 'Aroon', 'Trendmode', 'AD', 'ADOSC',
       'OBV', 'NATR', 'TRANGE','TWO_CROWS','THREE_BLACK_CROWS','THREE_INSIDE_UP_DOWN','CDL_3_LINE_STRIKE','CDL_3_OUTSIDE',
       'CDL_3_STARS_IN_SOUTH','CDL_3_WHITE_SOLDIERS','CDL_ABANDONED_BABY','CDL_ADVANCE_BLOCK','CDL_BELT_HOLD','CDL_BREAK_AWAY','CDL_CLOSING_MARUBOZU']


    plot_features =df[plot_cols]

    plot_features.index = df['time']

    df1 = plot_features


    df1= df1.astype(float)

    df1_time= df1.index


    df1.reset_index(drop=True, inplace=True)

    df1.describe().transpose()

   # BTCUSD =BTCUSD.loc[:23]

    

    column_indicies = {name: i for i, name in enumerate(df1.columns)}

    n =len(df1)
    #print(df1)
    #print(n)
    
    train_df =df1[0:int(n*0.7)]
   # print(train_df)
    val_df = df1[int(n*0.7): int(n*0.9)]
   # print(val_df)
    test_df = df1[int(n*0.9):]
   # print(test_df)
    
    num_features =df1.shape[1]

    train_mean= train_df.mean()
    train_std = train_df.std()
    predict_mean = predict_df.mean()
    predict_std = predict_df.std()


    
    scaled_cols =['time', 'change','change_price','open', 'high', 'low', 'close', 'vol_', 'volume', 'RSI', 'MACD',
       'macsignal', 'macdhist', 'ADX', 'Aroon', 'Trendmode', 'AD', 'ADOSC',
       'OBV', 'NATR', 'TRANGE']

    cat_cols = ['TWO_CROWS','THREE_BLACK_CROWS','THREE_INSIDE_UP_DOWN','CDL_3_LINE_STRIKE','CDL_3_OUTSIDE',
       'CDL_3_STARS_IN_SOUTH','CDL_3_WHITE_SOLDIERS','CDL_ABANDONED_BABY','CDL_ADVANCE_BLOCK','CDL_BELT_HOLD','CDL_BREAK_AWAY','CDL_CLOSING_MARUBOZU']


    train_df_scaled =pd.DataFrame(scaler.fit_transform(train_df[scaled_cols]), columns =scaled_cols )
    train_df_cat = train_df[cat_cols]
    train_df_cat.reset_index(inplace=True)
    train_df = train_df_scaled.merge(train_df_cat, left_index=True, right_index=True)
    

    val_df_scaled =pd.DataFrame(scaler.fit_transform(val_df[scaled_cols]), columns =scaled_cols )
    val_df_cat = val_df[cat_cols]
    val_df_cat.reset_index(inplace=True)
    val_df = val_df_scaled.merge(val_df_cat, left_index=True, right_index=True)
   # print(val_df)
    
    test_df_scaled =pd.DataFrame(scaler.fit_transform(test_df[scaled_cols]), columns =scaled_cols )
    test_df_cat = test_df[cat_cols]
    test_df_cat.reset_index(inplace=True)
    #test_df = pd.concat([test_df_scaled,test_df_cat ])
    test_df = test_df_scaled.merge(test_df_cat, left_index=True, right_index=True)

    predict_df_scaled =pd.DataFrame(scaler.fit_transform(predict_df[scaled_cols]), columns =scaled_cols )
    predict_df_cat = predict_df[cat_cols]
    predict_df_cat.reset_index(inplace=True)
    #predict_df = pd.concat([predict_df_scaled,predict_df_cat ])
    predict_df = predict_df_scaled.merge(predict_df_cat, left_index=True, right_index=True)
   # predict_df.drop(columns=['time'], axis=1, inplace=True)
   # predict_df['time'] = predict_time
    #print(predict_df.columns)
    #print('df:',predict_df)
    #print('time:',len(predict_time) )
    return train_df, val_df, test_df, predict_df, scaler , predict_mean, predict_std, train_mean, train_std


def create_time(df_, interval):
    strt = int(df_['new_time'].iloc[0])
    strt =datetime.utcfromtimestamp(strt).strftime('%Y-%m-%d %H:%M:%S')
    time_ = (pd.Series( data = pd.date_range(start=strt, periods=48, freq=interval)))
    return time_


def make_predict(coin,model,interval):
    #scaler = StandardScaler()
    #coin['change'] = coin['close'].pct_change(periods =24)
    #coin = coin.iloc[: , :]
    #coin= coin.dropna()
   
    
    #coin_price = coin[:]
    #df =BTCUSD[:-23]
    #coin_p =coin[-24:]
    train_df, val_df, test_df, predict_df, scaler, predict_mean,predict_std, train_mean,train_std = import_df(coin)
    
    
    last_price = predict_df['close'].iloc[-1]
    
    
    #print('inputs:',predict_df)
    coin_p =pd.DataFrame((predict_df), columns =predict_df.columns )
    #print('coin p:', np.array(coin_p))
    x= model.predict(np.array([coin_p,]))
    x =x[0][0:24,0]
   # x= np.argmax(x)
    coin_p['time'] 

    #print('inputs:',coin_p)
    #print('predictions:',)
    prediction = x * predict_std['close'] + predict_mean['close']
    last_price = last_price * train_std['close'] +train_mean['close']
    coin_p = coin_p * predict_std + predict_mean
    
    predict_df['new_time'] = predict_df['time'] #* predict_std['time'] + predict_mean['time']
    #predict_df = predict_df['new_time']
    time_ =create_time(predict_df, interval)
    #time_ = time_ * train_std['time'] + train_mean['time']
    #coin_p = x * train_std['change'] + train_mean['change']
    #prediction_change = prediction
    #print(last_price)
    #print(time_)
    
    return  prediction, coin_p, last_price, time_
    
    




########### Model Loads #########
#BTCUSD_5_m = tf.keras.models.load_model('models/BTCUSD/BTCUSD_5_m.h5')
BTCUSD_30_m = tf.keras.models.load_model('models/BTCUSD/BTCUSD_30m_LSTM.h5')
#BTCUSD_1_h = tf.keras.models.load_model('models/BTCUSD/BTCUSD_1_h.h5')
#################################
############# Predictions #######
####### BTCUSD ##################
def get_coin(pair, time):
    coin_ =cryptowatch1.get_ohlc(pair,time)
    model= tf.keras.models.load_model(f'models/{pair}/{pair}_{time}_LSTM.h5')
    coin_predict, coin_Prices,coin_LAST_PRICE, coin_Time = make_predict(coin_, model, time)
    time_list= ['5m', '15m','30m', '1h']
    for time_ in time_list:
        coin_24 = cryptowatch1.get_ohlc(pair,time_)
        coin_24['close'].to_csv(f'./trading-app/output_csv/{pair}/{pair}_{time_}_24h.csv', mode='w' )

    x=np.ravel(coin_predict)
    coin_predict = x[:24]
    print(coin_predict)
    coin_predict = coin_predict.tolist()
    coin_PRICES = coin_Prices['close'].tolist()
    print('coin_Prices:', coin_PRICES)
    predict_mean = round(mean(coin_predict),2)
    coin_PRICES.extend(coin_predict)
    outlist= coin_PRICES
    print(len(outlist))
    #print(outlist)
    coin_df = pd.DataFrame(coin_Time, columns=['new_time'])

   # print(coin_predict)
    coin_df['price'] = outlist 
    print('coin_df:',coin_df)
    #coin_df.to_csv(f'output_csv/BTCUSD/{pair}_{time}_LSTM.csv' )
    coin_df.to_csv(f'./trading-app/output_csv/{pair}/{pair}_{time}_LSTM.csv', mode='w' )

    return predict_mean



#################################
#get_coin('BTCUSD', '30m')
#get_coin('LTCUSD', '30m')
#get_coin('MATICUSD','30m')
#BTCUSD =cryptowatch1.get_ohlc('BTCUSD','30m')

asset_list=['ADAUSD','BTCUSD','DOGEUSD', 'ETHUSD',  'HFTUSD','LTCUSD', 'MATICUSD','SHIBUSD', 'SOLUSD', 'XRPUSD']

#asset_list=['BTCUSD']
for i in asset_list:
    try:
        get_coin(i,'5m')
        get_coin(i,'15m')
        get_coin(i,'30m')
        get_coin(i,'1h')
    except: print('not available')


#get_coin('BTCUSD','30m')
'''
def run_models():

    def five_m():
        for i in asset_list:
            get_coin(i, '5m')
    def fifteen_m():
        for i in asset_list:
            get_coin(i, '15m')
    def thirty_m():
        for i in asset_list:
            get_coin(i, '30m')
    def sixty_m():
        for i in asset_list:
            get_coin()
    
    
    schedule.every(5).minutes.do(five_m)
    get_coin
'''