import requests
from datetime import date
import datetime
import time
import json
import pandas as pd
from ta import *


date = datetime.datetime.now()
start_date = datetime.datetime(2022,1,1,20 )
#end_time = datetime.datetime(2022,10,2,20)
end_time = date
#date_time = datetime.datetime(2022,1,2,20)
unix_start =time.mktime(start_date.timetuple())
unix_end = time.mktime(end_time.timetuple())
unix_start=str(int(unix_start))
unix_end = str(int(unix_end))




def get_ohlc(pair, period):
    api_key= 'RD60460U1BEMOTECL3GN'
    period_dict= {'1m':60,'3m':180,'5m':300, '15m':900, '30m':1800, '1h':3600, '2h':7200, '4h':14400, '6h': 21600, '12h':43200, '1d': 86400, '3d':259200, '1w':604800}
    
    t =requests.get(f"https://api.cryptowat.ch/markets/KRAKEN/{pair}/ohlc?after={unix_start}&before{unix_end}&periods={period_dict[period]}&apikey={api_key}")
    response_dict = json.loads(t.text)
    #print(response_dict)

    
    r_l=response_dict['result'][f'{period_dict[period]}']


    df = pd.DataFrame(r_l)
    df.columns = ['time', 'open','high','low','close','vol_','volume']

    df['Date'] = pd.to_datetime(df['time'], 
                                  unit='s')

    df=df.drop("Date", axis =1)
    
    df_1 = indicators(df['high'],df['low'],df['close'],14,df['open'],df['volume'])

    df_RSI=df_1.RSI_()
    df["RSI"]=df_RSI

    df_MACD,df_macsignal, df_macdhist =df_1.MACD_()
    df['MACD'] = df_MACD
    df['macsignal'] =df_macsignal
    df['macdhist']= df_macdhist


    df_ADX = df_1.ADX_()
    df['ADX'] = df_ADX

    df_AROON =df_1.Aroon_()
    df['Aroon'] = df_AROON

    df_TRENDMODE =df_1.HT_TRENDMODE()
    df['Trendmode'] = df_TRENDMODE

    df_AD_ =df_1.AD_()
    df['AD'] = df_AD_

    df_ADOSC_ =df_1.ADOSC_()
    df['ADOSC'] = df_ADOSC_

    df_OBV_ =df_1.OBV_()
    df['OBV'] = df_OBV_

    df_NATR_ =df_1.NATR_()
    df['NATR'] = df_NATR_

    df_TRANGE_ =df_1.TRANGE_()
    df['TRANGE'] = df_TRANGE_

    df_TWO_CROWS= df_1.TWO_CROWS()
    df['TWO_CROWS']= df_TWO_CROWS

    df_THREE_BLACK_CROWS= df_1.THREE_BLACK_CROWS()
    df['THREE_BLACK_CROWS']= df_THREE_BLACK_CROWS

    df_THREE_INSIDE_UP_DOWN= df_1.THREE_INSIDE_UP_DOWN()
    df['THREE_INSIDE_UP_DOWN']= df_THREE_INSIDE_UP_DOWN

    df_CDL_3_LINE_STRIKE= df_1.CDL_3_LINE_STRIKE()
    df['CDL_3_LINE_STRIKE']= df_CDL_3_LINE_STRIKE

    df_CDL_3_OUTSIDE= df_1.CDL_3_OUTSIDE()
    df['CDL_3_OUTSIDE']= df_CDL_3_OUTSIDE

    df_CDL_3_STARS_IN_SOUTH= df_1.CDL_3_STARS_IN_SOUTH()
    df['CDL_3_STARS_IN_SOUTH']= df_CDL_3_STARS_IN_SOUTH


    df_CDL_3_WHITE_SOLDIERS= df_1.CDL_3_WHITE_SOLDIERS()
    df['CDL_3_WHITE_SOLDIERS']= df_CDL_3_WHITE_SOLDIERS

    df_CDL_ABANDONED_BABY= df_1.CDL_ABANDONED_BABY()
    df['CDL_ABANDONED_BABY']= df_CDL_ABANDONED_BABY

    df_CDL_ADVANCE_BLOCK= df_1.CDL_ADVANCE_BLOCK()
    df['CDL_ADVANCE_BLOCK']= df_CDL_ADVANCE_BLOCK

    df_CDL_BELT_HOLD= df_1.CDL_BELT_HOLD()
    df['CDL_BELT_HOLD']= df_CDL_BELT_HOLD

    df_CDL_BREAK_AWAY= df_1.CDL_BREAK_AWAY()
    df['CDL_BREAK_AWAY']= df_CDL_BREAK_AWAY

    df_CDL_CLOSING_MARUBOZU= df_1.CDL_CLOSING_MARUBOZU()
    df['CDL_CLOSING_MARUBOZU']= df_CDL_CLOSING_MARUBOZU

















    

    


    return df

#print(get_ohlc('BTCUSD', '30m'))

#df =get_ohlc('BTCUSD','30m')

#df.to_csv('test_test.csv')
#print(df)


#print(get_ohlc('BTCUSD', '30m'))