
import plotly
import plotly.graph_objects as go   
import plotly.express as px
import json
#from model_load import BTCUSD_30_M_df
import numpy as np
import pandas as pd
import plotly.figure_factory as ff







def prediction_plot(df_,mean_, coin_pair,time_):

    pre_price =df_[:24]
    prediction = df_[26:]
    #last_price = df_.iloc[25]
    #fig = []
    #last_price= pre_price.iloc[-1]
    fig = go.Figure(layout_title_text =f"{coin_pair} Closing Prices at {time_} Intervals", layout={'xaxis':{'title':'Time'},'yaxis':{'title':'Price in USD'}})
    fig.add_trace(go.Scatter( mode ='lines+markers',name='past 24 closing prices',text='previous closing prices', marker=dict(color='#2b5698'),x=pre_price['new_time'],y=pre_price['price']))
    #fig.add_trace(go.Scatter( mode ='lines+markers',text='previous closing prices', marker=dict(color='Blue'),x=pre_price['time'],y=pre_price['price'].iloc[-1]))
    #fig.add_trace(go.Scatter(x=last_price['time'], y=last_price['price']))
    #if last
    
    fig.add_trace(go.Scatter(mode ='markers',name='predicted closing prices',text='predicted closing prices',marker=dict(color='#b942b9' , symbol='arrow-right-open', size=15, angleref="previous"), x=prediction['new_time'],y=prediction['price']))
    fig.add_trace(go.Scatter(mode='lines', name='mean predicted closing price',marker = dict(color='#6efafd'),x=[prediction['new_time'].min(),prediction['new_time'].max()], y=prediction['price'] ))
    #fig.add_trace(go.line(df_[24:], x=df_['time'], y=df_['price']))
    #fig= px.scatter(prediction, x='time', y='price')
    

    fig.update_layout(template='plotly_dark', width =1000)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json





def violin_plot(df_5m,df_15m,df_30m,df_1h,close_price, predicted_price):

    #pre_price =df_[:24]
    #prediction = df_[26:]
    #last_price = df_.iloc[25]
    
    

    df_5m = df_5m[-24:]
    print('price',df_5m)
    df_5m['time'] = '5m' 

    df_15m = df_15m[-24:]
    df_15m['time'] = '15m'

    df_30m = df_30m[-24:]
    df_30m['time'] = '30m'

    df_1h = df_1h[-24:]
    df_1h['time'] = '1h'   
    print('price',df_1h)
    df_main= pd.concat([df_5m, df_15m],ignore_index=True)
    df_main= pd.concat([df_main, df_30m],ignore_index=True)
    df_main = pd.concat([df_main,df_1h],ignore_index=True)
    
    #df_list = [df_5m,df_15m,df_30m,df_1h]
    #df_24 = df_24['close'].tolist()

   

    group_labels =['close Prices']
    #df_24_mean = np.mean(df_24)
    #df_24_std = np.std(df_24)
    #low = df_24_mean + (-3 * df_24_std)
    #high = df_24_mean + (3 * df_24_std)
   # num_curve = np.linspace(start=low, stop=high, num= 1000 )
    #fig= ff.create_distplot([num_curve], group_labels , curve_type='normal',show_hist =False)
   
    #fig.add_trace(go.Scatter( 
    #text=["Line positioned relative to the plot",
    #      "Line positioned relative to the axes"],
    #mode="text",))
    fig = go.Figure()
    time_series = ['5m','15m','30m','1h']

    print(df_main)

    for time in time_series:
        fig.add_trace(go.Violin(x= df_main['time'][df_main['time']==time], y= df_main['close'][df_main['time']==time], name= 'Last 24 Closing Prices', box_visible=True, meanline_visible=True, points='all'))

    #fig.add_trace(go.shape)
    fig.add_hline(y=close_price, line = dict(color='#2b5698'), name='close price')
    fig.add_hline(y=predicted_price,line = dict(color='#6efafd'), name='predicted price')
    
   # print('df is 1h:',df_main[['close','time']][df_main['time']=='1h'])
   # print('df is 5m:',df_main[['close','time']][df_main['time']=='5m'])
    fig.update_yaxes(showticklabels=True)
    fig.update_layout(template='plotly_dark', width =1000)
    #fig.show()
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json



#df =pd.read_csv('BTCUSD_30M_df.csv')

#prediction_plot(df)
