import requests
import time
import schedule
import json
import numpy

import datetime 
import time
import pandas as pd

#################################################################
#################################################################
date_time = datetime.datetime(2021,7,26,20)
unix =time.mktime(date_time.timetuple())






################### Get OHLC ######################################
###################################################################

def get_ohlc(pair,interval):
    resp = requests.get(f'https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}&since{unix}')
    
    json_data =json.loads(resp.text)

    OHLC_LIST =[]
    OHLC_SET ={}
    time_ =[]
    open_ =[]
    high_ =[]
    low_ =[]
    close_ = []
    volume_ =[]

    for i in json_data['result'][f'{pair}']:
        OHLC_SET['time'] = i[0]
        time_.append(i[0])
        #print(OHLC_SET['time'])
        OHLC_SET['open'] =i[1]
        open_.append(i[1])
        
        OHLC_SET['high'] = i[2]
        high_.append(i[2])
        
        OHLC_SET['low'] = i[3]
        low_.append(i[3])
        
        OHLC_SET['close'] = i[4]
        close_.append(i[4])
        
        OHLC_SET['volume'] =i[6]
        volume_.append(i[6])
        
        OHLC_LIST.append(OHLC_SET)
    time_ = numpy.array(time_, dtype=float)
    open_ = numpy.array(open_, dtype=float)
    high_ =numpy.array(high_, dtype =float)
    low_ = numpy.array(low_, dtype=float)
    close_ = numpy.array(close_, dtype= float)
    volume_=numpy.array(volume_, dtype =float)
    OHLC_DF = pd.DataFrame({
        'time':time_,
        'open':open_,
        'high':high_,
        'low':low_,
        'close':close_,
        'volume':volume_

    })

   # return OHLC_LIST, time_, open_, high_, low_,close_,volume_,OHLC_DF
    #return OHLC_LIST
    return OHLC_DF
######################### XXBTZUSD########################
XXBTZUSD =get_ohlc('XXBTZUSD',5)
'''
XXBTZUSD_time = XXBTZUSD[1]
XXBTZUSD_open = XXBTZUSD[2]
XXBTZUSD_high = XXBTZUSD[3]
XXBTZUSD_low = XXBTZUSD[4]
XXBTZUSD_close = XXBTZUSD[5]
XXBTZUSD_volume = XXBTZUSD[6]
'''
#########################################################
#########################################################
#print(XXBTZUSD)