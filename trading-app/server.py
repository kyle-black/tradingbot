from flask import Flask, render_template, json
import requests
#from kraken_connection import KrakenConnection
#from kraken_connections import OHLC
#from model_load import *

#import model_load 
#from model_load import get_coin
import plotly_create
import pandas as pd 
import os
#from model_load import get_coin

app =Flask(__name__)

cwd = os.getcwd()

def getzscore(std_dev,price, mean):
   return round((price-mean) / std_dev,2)


def fetch_price(coin_pair, time_):
   
   pair = coin_pair
    
   time_series= ['5m', '15m', '30m', '1h']
   
   df_5m = pd.read_csv(f'output_csv/{pair}/{pair}_5m_24h.csv')
   df_15m = pd.read_csv(f'output_csv/{pair}/{pair}_15m_24h.csv')
   df_30m = pd.read_csv(f'output_csv/{pair}/{pair}_30m_24h.csv')
   df_1h = pd.read_csv(f'output_csv/{pair}/{pair}_1h_24h.csv')
   df_ = pd.read_csv(f'output_csv/{pair}/{pair}_{time_}_LSTM.csv')
   pre_ =df_[:24]
   pre_mean= pre_['price'].mean()
   predict_ = df_[26:]
   predict_mean = round(predict_['price'].mean(),)
   last_std = round(pre_['price'].std())
   last_closing_price = round(df_['price'].iloc[23],4)
   zscore =getzscore(last_std, last_closing_price, pre_mean  )
   #plot_ =plotly_create.prediction_plot(df_, predict_mean, coin_pair, time_)
   #violin_plot = plotly_create.violin_plot(df_5m,df_15m,df_30m,df_1h)
   last_closing_price = round(df_['price'].iloc[23],4)
   difference =round((100*(1-(last_closing_price/ predict_mean))),2)
   plot_ =plotly_create.prediction_plot(df_, predict_mean, coin_pair, time_)
   #violin_plot = plotly_create.violin_plot(df_5m,df_15m,df_30m,df_1h,close_price =last_closing_price,predicted_price=predict_mean)
   violin_plot =plotly_create.violin_plot(df_5m,df_15m,df_30m,df_1h,close_price=last_closing_price, predicted_price=predict_mean)
   context ={"PLOT":plot_}
   context2 = {"PLOT2":violin_plot}
      #print(cwd)

   #asset_dict ={f'{coin_pair}':{f'{time_}':{'context':context,'context2':context2,}}
   dict_pair = f'{coin_pair}_{time_}'
   
   asset_dict =dict({dict_pair :{'context':context,'context2':context2, 'last_closing_price':last_closing_price, 'predict_mean':predict_mean,'difference':difference, 'last_std':last_std, 'zscore':zscore}} )


   return asset_dict, difference






@app.route('/')
def route_create():
   coins_ = ['BTCUSD', 'ADAUSD', 'DOGEUSD', 'ETHUSD','LTCUSD','MATICUSD','SHIBUSD','SOLUSD','XRPUSD']
   time_series= ['5m', '15m', '30m', '1h']

   
   BTCUSD_5m, BTCUSD_5m_diff =fetch_price('BTCUSD', '5m')
   BTCUSD_5m,BTCUSD_15m_diff =fetch_price('BTCUSD', '15m')
   BTCUSD_30m,BTCUSD_30m_diff =fetch_price('BTCUSD', '30m')
   BTCUSD_1h,BTCUSD_1h_diff =fetch_price('BTCUSD', '1h')

   ADAUSD_5m, ADAUSD_5m_diff =fetch_price('ADAUSD', '5m')
   ADAUSD_5m,ADAUSD_15m_diff =fetch_price('ADAUSD', '15m')
   ADAUSD_30m,ADAUSD_30m_diff =fetch_price('ADAUSD', '30m')
   ADAUSD_1h,ADAUSD_1h_diff =fetch_price('ADAUSD', '1h')

   DOGEUSD_5m, DOGEUSD_5m_diff =fetch_price('DOGEUSD', '5m')
   DOGEUSD_5m,DOGEUSD_15m_diff =fetch_price('DOGEUSD', '15m')
   DOGEUSD_30m,DOGEUSD_30m_diff =fetch_price('DOGEUSD', '30m')
   DOGEUSD_1h,DOGEUSD_1h_diff =fetch_price('DOGEUSD', '1h')

   DOGEUSD_5m, DOGEUSD_5m_diff =fetch_price('DOGEUSD', '5m')
   DOGEUSD_5m,DOGEUSD_15m_diff =fetch_price('DOGEUSD', '15m')
   DOGEUSD_30m,DOGEUSD_30m_diff =fetch_price('DOGEUSD', '30m')
   DOGEUSD_1h,DOGEUSD_1h_diff =fetch_price('DOGEUSD', '1h')

   ETHUSD_5m, ETHUSD_5m_diff =fetch_price('ETHUSD', '5m')
   ETHUSD_5m,ETHUSD_15m_diff =fetch_price('ETHUSD', '15m')
   ETHUSD_30m,ETHUSD_30m_diff =fetch_price('ETHUSD', '30m')
   ETHUSD_1h,ETHUSD_1h_diff =fetch_price('ETHUSD', '1h')

  


   LTCUSD_5m, LTCUSD_5m_diff =fetch_price('LTCUSD', '5m')
   LTCUSD_5m,LTCUSD_15m_diff =fetch_price('LTCUSD', '15m')
   LTCUSD_30m,LTCUSD_30m_diff =fetch_price('LTCUSD', '30m')
   LTCUSD_1h,LTCUSD_1h_diff =fetch_price('LTCUSD', '1h')


   MATICUSD_5m, MATICUSD_5m_diff =fetch_price('MATICUSD', '5m')
   MATICUSD_5m,MATICUSD_15m_diff =fetch_price('MATICUSD', '15m')
   MATICUSD_30m,MATICUSD_30m_diff =fetch_price('MATICUSD', '30m')
   MATICUSD_1h,MATICUSD_1h_diff =fetch_price('MATICUSD', '1h')


   SHIBUSD_5m_diff =fetch_price('SHIBUSD', '5m')
   SHIBUSD_5m,SHIBUSD_15m_diff =fetch_price('SHIBUSD', '15m')
   SHIBUSD_30m,SHIBUSD_30m_diff =fetch_price('SHIBUSD', '30m')
   SHIBUSD_1h,SHIBUSD_1h_diff =fetch_price('SHIBUSD', '1h')


   SOLUSD_5m_diff =fetch_price('SOLUSD', '5m')
   SOLUSD_5m,SOLUSD_15m_diff =fetch_price('SOLUSD', '15m')
   SOLUSD_30m,SOLUSD_30m_diff =fetch_price('SOLUSD', '30m')
   SOLUSD_1h,SOLUSD_1h_diff =fetch_price('SOLUSD', '1h')

  

   BTCUSD_5m, BTCUSD_5m_diff =fetch_price('BTCUSD', '5m')
   BTCUSD_5m,BTCUSD_15m_diff =fetch_price('BTCUSD', '15m')
   BTCUSD_30m,BTCUSD_30m_diff =fetch_price('BTCUSD', '30m')
   BTCUSD_1h,BTCUSD_1h_diff =fetch_price('BTCUSD', '1h')















   diff_dict ={'BTCUSD_5m':BTCUSD_5m_diff,'BTCUSD_15m':BTCUSD_15m_diff,'BTCUSD_30m':BTCUSD_30m_diff,'BTCUSD_1h':BTCUSD_1h_diff, 'ADAUSD_5m':ADAUSD_5m_diff,'ADAUSD_15m':ADAUSD_15m_diff,'ADAUSD_30m':ADAUSD_30m_diff,'ADAUSD_1h':ADAUSD_1h_diff, 'DOGEUSD_5m':DOGEUSD_5m_diff,'DOGEUSD_15m':DOGEUSD_15m_diff,'DOGEUSD_30m':DOGEUSD_30m_diff,'DOGEUSD_1h':DOGEUSD_1h_diff, 'ETHUSD_5m':ETHUSD_5m_diff,'ETHUSD_15m':ETHUSD_15m_diff, 'ETHUSD_30m':ETHUSD_30m_diff,'ETHUSD_1h':ETHUSD_1h_diff,'LTCUSD_5m':LTCUSD_5m_diff,'LTCUSD_15m':LTCUSD_15m_diff,'LTCUSD_30m':LTCUSD_30m_diff,'LTCUSD_1h':LTCUSD_1h_diff, 'MATICUSD_5m':MATICUSD_5m_diff, 'MATICUSD_15m':MATICUSD_15m_diff,'MATICUSD_30m':MATICUSD_30m_diff,'MATICUSD_1h':MATICUSD_1h_diff, 'SHIBUSD_5m':SHIBUSD_1h_diff,'SHIBUSD_15m':SHIBUSD_15m_diff,'SHIBUSD_30m':SHIBUSD_30m_diff,'SHIBUSD_1h':SHIBUSD_1h_diff,'SOLUSD_5m':SOLUSD_5m_diff,'SOLUSD_15m':SOLUSD_15m_diff,'SOLUSD_30m':SOLUSD_30m_diff,'SOLUSD_1h':SOLUSD_1h_diff }
  # return diff_dict
   return render_template('index5.html', diff_dict=diff_dict)
  
   




@app.route('/coin/<coin_pair>/<time_>')
def coin_(coin_pair,time_):
   
   #context,context2, last_closing_price, predict_mean,difference, last_std, zscore = fetch_price(coin_pair, time_)
   asset_dict, asset_diff= fetch_price(coin_pair, time_) 
   dict_pair = f'{coin_pair}_{time_}'
   context = asset_dict[dict_pair]['context']
   context2 = asset_dict[dict_pair]['context2']
   last_closing_price = asset_dict[dict_pair]['last_closing_price']
   prediction_mean = asset_dict[dict_pair]['predict_mean']
   difference = asset_dict[dict_pair]['difference']
   last_std = asset_dict[dict_pair]['last_std']
   zscore = asset_dict[dict_pair]['zscore']
   diff_dict =route_create()

   
   return render_template("plotly.html", context=context, context2=context2,last_closing_price = last_closing_price,prediction_mean=prediction_mean, difference =difference, last_std=last_std, zscore =zscore, time_=time_,diff_dict=diff_dict)

@app.route('/how/methodology')
def how_to():
   diff_dict =route_create()
   return render_template('explain.html', diff_dict=diff_dict)


#@app.route('/coin/<coin_pair>')
#def main_coin(coin_pair):
#   return render_template("plotly.html", context=context,last_closing_price = last_closing_price,prediction_mean=prediction_mean)
