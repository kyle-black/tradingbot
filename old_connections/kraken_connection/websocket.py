# Import required Python libraries
import time
import base64
import hashlib
import hmac
import urllib.request
import json
import keys
 
# Copy/paste API public key and API private key as displayed in account management
api_key = keys.API_KEY
api_secret = keys.PRIVATE_KEY
 
# Variables (API method, nonce, and POST data)
api_path = '/0/private/GetWebSocketsToken'
api_nonce = str(int(time.time()*1000))
api_post = 'nonce=' + api_nonce
 
# Cryptographic hash algorithms
api_sha256 = hashlib.sha256(api_nonce.encode('utf-8') + api_post.encode('utf-8'))
api_hmac = hmac.new(base64.b64decode(api_secret), api_path.encode('utf-8') + api_sha256.digest(), hashlib.sha512)
 
# Encode signature into base64 format used in API-Sign value
api_signature = base64.b64encode(api_hmac.digest())
 
# HTTP request (POST)
api_request = urllib.request.Request('https://api.kraken.com/0/private/GetWebSocketsToken', api_post.encode('utf-8'))
api_request.add_header('API-Key', api_key)
api_request.add_header('API-Sign', api_signature)
api_response = urllib.request.urlopen(api_request).read().decode()

# Output API response
print(json.loads(api_response)['result']['token'])
web_socket = json.loads(api_response)['result']['token']