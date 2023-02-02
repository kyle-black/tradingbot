from kraken_connection import KrakenConnection


myhandler =KrakenConnection.my_handler(message)

KrakenConnection.my_client.subscribe_public(
    subscription ={
        'name': 'trade'
    },
    pair =['XBT/USD', 'XRP/USD'],
    callback = myhandler
)