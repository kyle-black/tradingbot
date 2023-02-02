import talib
import numpy
from OHLC import *
import pandas as pd




############### Traditional Indicators ############################

class indicators:
    def __init__(self,high_,low_,close_, time_period, open_, volume_):
        #self.price_data = price_data
        self.high = high_
        self.low =low_
        self.close = close_
        self.time_period = time_period
        self.open = open_
        self.volume = volume_
    
    def RSI_(self):
        RSI =talib.RSI(self.close, self.time_period)
        return RSI
    def MACD_(self,**kwargs):
        MACD,macsignal,macdhist = talib.MACD(self.close, self.time_period)
        return MACD, macsignal , macdhist
    def ADX_(self):
        ADX = talib.ADX(self.high,self.low,self.close)
        return ADX
    def Aroon_(self):
        Aroon = talib.AROONOSC(self.high, self.low)
        return Aroon
    def HT_TRENDMODE(self):
        HT_TREND =talib.HT_TRENDMODE(self.close)
        return HT_TREND
    def AD_(self):
        AD = talib.AD(self.high, self.low,self.close,self.volume)
        return AD
    def ADOSC_(self):
        ADOSC = talib.ADOSC(self.high, self.low,self.close,self.volume, fastperiod=3,slowperiod=10)
        return ADOSC
    def OBV_(self):
        OBV = talib.OBV(self.close,self.volume)
        return OBV

    def NATR_(self):
        NATR= talib.NATR(self.high,self.low,self.close,timeperiod=14)
        return NATR
    def TRANGE_(self):
        TRANGE = talib.TRANGE(self.high,self.low,self.close)
        return TRANGE
    def TWO_CROWS(self):
        return talib.CDL2CROWS(self.open,self.high,self.low,self.close)
    def THREE_BLACK_CROWS(self):
        return talib.CDL3BLACKCROWS(self.open,self.high,self.low,self.close)
    def THREE_INSIDE_UP_DOWN(self):
        return talib.CDL3INSIDE(self.open,self.high,self.low,self.close)
    def CDL_3_LINE_STRIKE(self):
        return talib.CDL3LINESTRIKE(self.open,self.high,self.low,self.close)
    def CDL_3_OUTSIDE(self):
        return talib.CDL3OUTSIDE(self.open,self.high,self.low,self.close)
    def CDL_3_STARS_IN_SOUTH(self):
        return  talib.CDL3STARSINSOUTH(self.open,self.high,self.low,self.close)
    def CDL_3_WHITE_SOLDIERS(self):
        return talib.CDL3WHITESOLDIERS(self.open,self.high,self.low,self.close)
    def CDL_ABANDONED_BABY(self):
        return talib.CDLABANDONEDBABY(self.open, self.high, self.low, self.close, penetration=0)
    def CDL_ADVANCE_BLOCK(self):
        return talib.CDLADVANCEBLOCK(self.open,self.high,self.low,self.close)
    def CDL_BELT_HOLD(self):
        return talib.CDLBELTHOLD(self.open,self.high,self.low,self.close)
    def CDL_BREAK_AWAY(self):
        return talib.CDLBREAKAWAY(self.open,self.high,self.low,self.close)
    def CDL_CLOSING_MARUBOZU(self):
        return talib.CDLCLOSINGMARUBOZU(self.open,self.high,self.low,self.close)

    
#################################################################
#################################################################
class pattern(indicators):
    
    def TWO_CROWS(self):
        return talib.CDL2CROWS(self.open,self.high,self.low,self.close)
    def THREE_BLACK_CROWS(self):
        return talib.CDL3BLACKCROWS(self.open,self.high,self.low,self.close)
    def THREE_INSIDE_UP_DOWN(self):
        return talib.CDL3INSIDE(self.open,self.high,self.low,self.close)
    def CDL_3_LINE_STRIKE(self):
        return talib.CDL3LINESTRIKE(self.open,self.high,self.low,self.close)
    def CDL_3_OUTSIDE(self):
        return talib.CDL3OUTSIDE(self.open,self.high,self.low,self.close)
    def CDL_3_STARS_IN_SOUTH(self):
        return  talib.CDL3STARSINSOUTH(self.open,self.high,self.low,self.close)
    def CDL_3_WHITE_SOLDIERS(self):
        return talib.CDL3WHITESOLDIERS(self.open,self.high,self.low,self.close)
    def CDL_ABANDONED_BABY(self):
        return talib.CDLABANDONEDBABY(self.open, self.high, self.low, self.close, penetration=0)
    def CDL_ADVANCE_BLOCK(self):
        return talib.CDLADVANCEBLOCK(self.open,self.high,self.low,self.close)
    def CDL_BELT_HOLD(self):
        return talib.CDLBELTHOLD(self.open,self.high,self.low,self.close)
    def CDL_BREAK_AWAY(self):
        return talib.CDLBREAKAWAY(self.open,self.high,self.low,self.close)
    def CDL_CLOSING_MARUBOZU(self):
        return talib.CDLCLOSINGMARUBOZU(self.open,self.high,self.low,self.close)
   
class forward():
    def __init__(self, type_input, close, high, type_, advance):
       
        self.type_input = type_input
        self.close =close
        self.high =high
        self.type = type_
        self.advance = advance
    def count_intstances(self):
        if self.type == "RSI":
            RSI_IDX = (self.type_input<30).nonzero()
        return len(RSI_IDX[0])
    
    def lookahead_(self):
        input_length =len(self.type_input)
        if self.type =="RSI":
            RSI_input =self.type_input
            RSI_input = (RSI_input<30).nonzero()
            RSI_COUNT = len(RSI_input)
            Base = RSI_input[0]

        if self.type == "MACD":
            MACD_input =self.type_input
            MACD_input = np.where(MACD_input >0)[0] +1 


        price_list =[]
        segment_dict ={}
        change_dict ={}
        for idx, val in enumerate(Base):
        
            #### Look forward 60
            fwd_idx =val + self.advance
            ####### RSI Trigger INDEX
       
            Trigger = self.close[val]
            ##########3 Next 60 Price Closing
            if fwd_idx < input_length:
                Next_prices = self.high[val:fwd_idx]
                
                segment_dict[val] =Next_prices
                #####################################3

                change = numpy.round_(1-(Trigger / Next_prices), 5)
                change_dict[val] = change 

                
        #######################################
            

        return   change_dict







##################################################################

#ind = indicators(c,c,14)
#XXBTZUSD = indicators(XXBTZUSD_high,XXBTZUSD_low,XXBTZUSD_close,14, XXBTZUSD_open, XXBTZUSD_volume)
'''
XXBTZUSD = indicators(XXBTZUSD_high,XXBTZUSD_low,XXBTZUSD_close,14, XXBTZUSD_open, XXBTZUSD_volume)
input_MACD = XXBTZUSD.MACD_()
#t =forward(input_RSI, XXBTZUSD_close,XXBTZUSD_high, 'RSI', 60)
r = t.lookahead_RSI()
print(r)
'''