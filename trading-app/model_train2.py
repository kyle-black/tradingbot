import os


import tensorflow as tf
os.environ['TF_CCP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras

print(tf.version.VERSION)


import os
import datetime
import warnings

import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.preprocessing import StandardScaler

import cryptowatch1
#from dataparse import BTCUSD
warnings.filterwarnings('ignore')



#Bpd.read_csv('test_test.csv')


def import_df(df_):

    df_['change'] = df_['close'].pct_change(periods =24)
    df_['change_price'] = df_['close'].shift(24) + (df_['change'] * df_['close'].shift(24))
    df_= df_.dropna()

    df =df_[:-24]
    predict_df =df_[-24:]


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


    train_mean = predict_df.mean()
    train_std = predict_df.std()

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
    print(val_df)
    
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
    
    return train_df, val_df, test_df, predict_df, scaler , train_mean, train_std



#train_df,val_df,test_df,predict_df, scaler, train_mean, train_std =import_df(BTCUSD)

#import_df(BTCUSD)
#print(train_df)
#print(val_df)



class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    
   
    
    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])


def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window



def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset


@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)



@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example


#OUT_STEPS = 48
#multi_window = WindowGenerator(input_width=24,
#                               label_width=OUT_STEPS,
#                               shift=OUT_STEPS, label_columns=['close'])



def create_model(window, num_features):
    MAX_EPOCHS =20
    OUT_STEPS=24
    CONV_WIDTH = 8
    num_features= num_features
    '''
    model =tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])
    '''

    model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features].
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
        
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=2,
                                                    mode='min')

    model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])
    model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
        
        
    
    return model





coin_list=['BTCUSD','ETHUSD','SOLUSD', 'MATICUSD','BNBUSD','DOGEUSD','XRPUSD','HFTUSD','LTCUSD','ADAUSD','SHIBUSD']
times=['5m','15m','30m','1h']

def save_models(coin_list,times):
 # create_model = WindowGenerator()

  for coin in coin_list:
    for time in times:
       
      
      
        try:
            coin_ = cryptowatch1.get_ohlc(coin,time)
            train_df,val_df,test_df,predict_df, scaler, train_mean, train_std =import_df(coin_)
            num_features =train_df.shape[1]
            OUT_STEPS = 24
            multi_window = WindowGenerator(input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS, label_columns=['close'],train_df=train_df, val_df=val_df, test_df=test_df)
            
            model = create_model(multi_window, num_features)
            model.save(f'models/{coin}/{coin}_{time}_LSTM.h5')
        except: print(f'No {coin} and/or {time}')
    


save_models(coin_list,times)















'''
BTCUSD1 = BTCUSD_p


p_mean = BTCUSD1.mean()
p_std = BTCUSD1.std()


BTCUSD =pd.DataFrame(scaler.fit_transform(BTCUSD1), columns =df1.columns )

x=model.predict(np.array([BTCUSD,]))


x = x * p_std['close'] + p_mean['close']


#model.save('models/BTCUSD/BTCUSD_30_m.h5')
#model.save('models/BTCUSD/BTCUSD_1_h.h5')
model.save('models/BTCUSD/BTCUSD_5_m.h5')


x=np.ravel(x)
predict_5_min = x
predict_5_min = predict_5_min.tolist()
#print(x)
'''