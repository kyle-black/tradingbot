from kraken_wsclient_py import kraken_wsclient_py as client
#from websocket import web_socket



#ws_token = web_socket 

ws_token ='jtSty8KqcD8yy3FuLJjO8A2B6cMOrGjOFgBSwQ8Te7c'


def my_handler(message):
    # Here you can do stuff with the messages
    try:
        parse = message[1][0][0]
        print("this is the parsed transaction:" + parse)
        return parse
    except:    
        None
    

my_client = client.WssClient()
my_client.start()

# Sample public-data subscription:

my_client.subscribe_public(
    subscription = {
        'name': 'trade'
    },
    pair = ['XBT/USD', 'XRP/USD'],
    callback = my_handler
)

#my_handler()


# Sample private-data subscription:
'''
my_client.subscribe_private(
    subscription = {
        'name': 'openOrders',
        'token': ws_token
    },
    callback = my_handler
)
'''
'''
my_client.subscribe_private(
    subscription = {
        'name': 'ownTrades',
        'token': ws_token
    },
    callback = my_handler
)
'''
# Sample order-entry call:
'''
my_client.request(
    request = {
        'token': ws_token,
        'event': 'addOrder',
        'type': 'buy',
        'ordertype': 'limit',
        'pair': 'XBT/USD',
        'price': '9000',
        'volume': '0.01',
        'userref': '666'
    },
    callback = my_handler
)
'''