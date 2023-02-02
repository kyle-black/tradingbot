import krakenWS as cfWsApi
import util.cfLogging as cfLog
logger = cfLog.CfLogger.get_logger(" Example ")
import keys

########################################################################################################################
# Please insert you API key and secret
########################################################################################################################

#api_path = "wss://www.cryptofacilities.com/ws/v1"
api_path= "wss://futures.kraken.com/ws/v1" 
api_key = keys.API_KEY  # accessible on your Account page under Settings -> API Keys
api_secret = keys.PRIVATE_KEY  # accessible on your Account page under Settings -> API Keys
timeout = 10
trace = False  # set to True for connection verbose logging

cfWs = cfWsApi.CfWebSocketMethods(base_url=api_path, api_key=api_key, api_secret=api_secret, timeout=10, trace=trace)


def subscribe_api_tester():
    """Test the subscribe methods"""

    ##### public feeds #####
'''
    product_ids = ["PI_XBTUSD"]

    # subscribe to trade
    feed = "trade"
    cfWs.subscribe_public(feed, product_ids)

    # subscribe to book
    feed = "book"
    cfWs.subscribe_public(feed, product_ids)

    # subscribe to ticker
    feed = "ticker"
    cfWs.subscribe_public(feed, product_ids)

    # subscribe to ticker lite
    feed = "ticker_lite"
    cfWs.subscribe_public(feed, product_ids)

    # subscribe to heartbeat
    feed = "heartbeat"
    cfWs.subscribe_public(feed)
'''

    ##### private feeds #####


    # subscribe to account balances and margis
feed = "account_balances_and_margins"
cfWs.subscribe_private(feed)
'''
        # subscribe to account log
    feed = "account_log"
    cfWs.subscribe_private(feed)

        # subscribe to deposits withdrawals
    feed = "deposits_withdrawals"
    cfWs.subscribe_private(feed)

        # subscribe to fills
    feed = "fills"
    cfWs.subscribe_private(feed)

        # subscribe to open positions
    feed = "open_positions"
    cfWs.subscribe_private(feed)

        # subscribe to open orders
    feed = "open_orders"
    cfWs.subscribe_private(feed)

        # subscribe to notifications
    feed = "notifications_auth"
    cfWs.subscribe_private(feed)

'''
def unsubscribe_api_tester():
    """Test the unsubscribe methods"""

    ##### public feeds #####

    product_ids = ["FV_XRPXBT_180615"]

    # unsubscribe to trade
    feed = "trade"
    cfWs.unsubscribe_public(feed, product_ids)

    # unsubscribe to book
    feed = "book"
    cfWs.unsubscribe_public(feed, product_ids)

    # unsubscribe to ticker
    feed = "ticker"
    cfWs.unsubscribe_public(feed, product_ids)

    # unsubscribe to ticker lite
    feed = "ticker_lite"
    cfWs.unsubscribe_public(feed, product_ids)

    # unsubscribe to heartbeat
    feed = "heartbeat"
    cfWs.unsubscribe_public(feed)

    ##### private feeds #####

    # unsubscribe to account balances and margins
    feed = "account_balances_and_margins"
    cfWs.unsubscribe_private(feed)

    # unsubscribe to account log
    feed = "account_log"
    cfWs.unsubscribe_private(feed)

    # unsubscribe to deposits withdrawals
    feed = "deposits_withdrawals"
    cfWs.unsubscribe_private(feed)

    # unsubscribe to fills
    feed = "fills"
    cfWs.unsubscribe_private(feed)

    # unsubscribe to open positions
    feed = "open_positions"
    cfWs.unsubscribe_private(feed)

    # unsubscribe to open orders
    feed = "open_orders"
    cfWs.unsubscribe_private(feed)

    # unsubscribe to notifications
    feed = "notifications_auth"
    cfWs.unsubscribe_private(feed)


logger.info("-----------------------------------------------------------")
logger.info("****PRESS ANY KEY TO SUBSCRIBE AND START RECEIVING INFO****")
logger.info("-----------------------------------------------------------")
input()

# Subscribe
subscribe_api_tester()
logger.info("-----------------------------------------------------------")
logger.info("****PRESS ANY KEY TO UNSUBSCRIBE AND EXIT APPLICATION****")
logger.info("-----------------------------------------------------------")
input()

# Unsubscribe
unsubscribe_api_tester()

# Exit
exit()

