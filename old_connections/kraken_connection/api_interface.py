from websocket import create_connection 
#from websocket import create_connection
ws = create_connection("wss://ws.kraken.com/")

ws.send('{"event":"subscribe", "subscription":{"name":"trade"}, "pair":["XBT/USD","XRP/USD"]}')

while True:

    print(ws.recv())
'''
def on_message(wsapp, message):
    print(message)

wsapp = websocket.WebSocketApp("wss://ws.kraken.com/", subprotocols=["STOMP"], on_message=on_message)
wsapp.run_forever()
'''
