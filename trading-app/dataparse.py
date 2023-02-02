from ta import *
from cryptowatch1 import BTCUSD
import pandas as pd
import os













############################   XXBTZUSD ##################################################3
####### RSI ###############################################################################
'''
XXBTZUSD = indicators(XXBTZUSD_high,XXBTZUSD_low,XXBTZUSD_close,14, XXBTZUSD_open, XXBTZUSD_volume)
XXBTZUSD_RSI = XXBTZUSD.RSI_()
XXBTZUSD_forward_RSI =forward(XXBTZUSD_RSI, XXBTZUSD_close,XXBTZUSD_high, 'RSI', 60)
XXBTZUSD_change_RSI = XXBTZUSD_forward_RSI.lookahead_()
'''
####### MACD ##############################################################################

#XXBTZUSD = indicators(XXBTZUSD_high,XXBTZUSD_low,XXBTZUSD_close,14, XXBTZUSD_open, XXBTZUSD_volume)
#XXBTZUSD_MACD = XXBTZUSD.MACD_()
#XXBTZUSD_forward_MACD =forward(XXBTZUSD_MACD, XXBTZUSD_close,XXBTZUSD_high, 'MACD', 60)
#XXBTZUSD_change_MACD = XXBTZUSD_forward_MACD.lookahead_()


###########################################################
BTC_USD = indicators(BTCUSD['high'],BTCUSD['low'],BTCUSD['close'],14,BTCUSD['open'],BTCUSD['volume'])

BTC_RSI=BTC_USD.RSI_()
BTCUSD["btc_RSI"]=BTC_RSI

BTC_MACD,BTC_macsignal, BTC_macdhist = BTC_USD.MACD_()
BTCUSD['btc_MACD'] = BTC_MACD
BTCUSD['btc_macsignal'] =BTC_macsignal
BTCUSD['btc_macdhist']= BTC_macdhist


BTC_ADX = BTC_USD.ADX_()
BTCUSD['btc_ADX'] = BTC_ADX

BTC_AROON =BTC_USD.Aroon_()
BTCUSD['btc_Aroon'] = BTC_AROON

BTC_TRENDMODE =BTC_USD.HT_TRENDMODE()
BTCUSD['btc_Trendmode'] = BTC_TRENDMODE

BTC_AD_ =BTC_USD.AD_()
BTCUSD['btc_AD'] = BTC_AD_

BTC_ADOSC_ =BTC_USD.ADOSC_()
BTCUSD['btc_ADOSC'] = BTC_ADOSC_

BTC_OBV_ =BTC_USD.OBV_()
BTCUSD['btc_OBV'] = BTC_OBV_

BTC_NATR_ =BTC_USD.NATR_()
BTCUSD['btc_NATR'] = BTC_NATR_

BTC_TRANGE_ =BTC_USD.TRANGE_()
BTCUSD['btc_TRANGE'] = BTC_TRANGE_








#########################################################################3
'''
BTC_USD = pattern(BTCUSD['high'],BTCUSD['low'],BTCUSD['close'],14,BTCUSD['open'],BTCUSD['volume'])



btc_two_crows =BTC_USD.TWO_CROWS()
BTCUSD['btc_two_crows'] = btc_two_crows

btc_THREE_BLACK_CROWS=BTC_USD.THREE_BLACK_CROWS()
BTCUSD['btc_THREE_BLACK_CROWS'] = btc_THREE_BLACK_CROWS

btc_THREE_INSIDE_UP_DOWN=BTC_USD.THREE_INSIDE_UP_DOWN()
BTCUSD['btc_THREE_INSIDE_UP_DOWN'] =btc_THREE_INSIDE_UP_DOWN

btc_CDL_3_LINE_STRIKE=BTC_USD.CDL_3_LINE_STRIKE()
BTCUSD['CDL_3_LINE_STRIKE'] =btc_CDL_3_LINE_STRIKE

btc_CDL_3_OUTSIDE=BTC_USD.CDL_3_OUTSIDE()
BTCUSD['CDL_3_OUTSIDE'] =btc_CDL_3_OUTSIDE

btc_CDL_3_STARS_IN_SOUTH=BTC_USD.CDL_3_STARS_IN_SOUTH()
BTCUSD['CDL_3_STARS_IN_SOUTH'] =btc_CDL_3_STARS_IN_SOUTH

btc_CDL_3_WHITE_SOLDIERS=BTC_USD.CDL_3_WHITE_SOLDIERS()
BTCUSD['CDL_3_WHITE_SOLDIERS'] =btc_CDL_3_WHITE_SOLDIERS

btc_CDL_ABANDONED_BABY=BTC_USD.CDL_ABANDONED_BABY()
BTCUSD['CDL_ABANDONED_BABY'] =btc_CDL_ABANDONED_BABY

btc_CDL_ADVANCE_BLOCK=BTC_USD.CDL_ADVANCE_BLOCK()
BTCUSD['CDL_ADVANCE_BLOCK'] =btc_CDL_ADVANCE_BLOCK

btc_CDL_BELT_HOLD =BTC_USD.CDL_BELT_HOLD()
BTCUSD['btc_CDL_BELT_HOLD'] =btc_CDL_BELT_HOLD

btc_CDL_BREAK_AWAY =BTC_USD.CDL_BREAK_AWAY()
BTCUSD['btc_CDL_BREAK_AWAY'] =btc_CDL_BREAK_AWAY

btc_CDL_CLOSING_MARUBOZU =BTC_USD.CDL_CLOSING_MARUBOZU()
BTCUSD['btc_CDL_CLOSING_MARUBOZU'] =btc_CDL_CLOSING_MARUBOZU


BTCUSD =BTCUSD

#print(btc_CDL_3_OUTSIDE)

BTCUSD.to_csv('btc_5min_test.csv')
'''
#print(XXBTZUSD_MACD)
print(BTCUSD)
BTCUSD.to_csv('btc_new.csv')