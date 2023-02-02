import krakenWS as cfWsApi


import util.cfLogging as cfLog
logger = cfLog.CfLogger.get_logger(" Example ")
import keys
########################################################################################################################
# Please insert you API key and secret
########################################################################################################################

api_path= "wss://futures.kraken.com/ws/v1" 
api_key = keys.API_KEY  # accessible on your Account page under Settings -> API Keys
api_secret = keys.PRIVATE_KEY  # accessible on your Account page under Settings -> API Keys
timeout = 10
trace = False  # set to True for connection verbose logging

cfWs = cfWsApi.CfWebSocketMethods(base_url=api_path, api_key=api_key, api_secret=api_secret, timeout=10, trace=trace)





def subscribe_account_log():

    feed= "deposits_withdrawls"
    return cfWs.subscribe_private(feed)


def unsubscribe_account_log():
    feed ="deposits_withdrawls"
    return cfWs.subscribe_private(feed)



logger.info("-----------------------------------------------------------")
logger.info("****PRESS ANY KEY TO SUBSCRIBE AND START RECEIVING INFO****")
logger.info("-----------------------------------------------------------")
input()

subscribe_account_log()
logger.info("-----------------------------------------------------------")
logger.info("****PRESS ANY KEY TO UNSUBSCRIBE AND EXIT APPLICATION****")
logger.info("-----------------------------------------------------------")
input()


unsubscribe_account_log()

# Exit
exit()