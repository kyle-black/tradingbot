//Global Variables
var globalIsWebSocketOpen = false;

//open websocket connection
function openWebSocket() 
{ 
  //First test for the browsers support for WebSockets
  if (!window.WebSocket) 
  {
    //If the user's browser does not support WebSockets, give an alert message
    alert("Your browser does not support the WebSocket API!");
  } 
  else 
  {
    //Set the websocket server URL
    var websocketurl = "wss://ws.kraken.com/";

    //get status element
    var connstatus = document.getElementById("connectionstatus");

    //get info div element
    var infodiv = document.getElementById("info");

    //Create the WebSocket object (web socket echo test service provided by websocket.org)
    socket = new WebSocket(websocketurl);

    //This function is called when the websocket connection is opened
    socket.onopen = function() 
    {
      outputConsoleMessage("Connected! :)")
      globalIsWebSocketOpen = true;
    };

    //This function is called when the websocket connection is closed
    socket.onclose = function() 
    {
      outputConsoleMessage("Disconnected from the websocket server at: " + websocketurl)
      globalIsWebSocketOpen = false;
    };

    //This function is called when the websocket receives a message. It is passed the message object as its only parameter
    socket.onmessage = function(message) 
    {
      outputConsoleMessage(message.data)
    };
  }
}

//function to send a message to the websocket server
function sendWebSocketMessage() 
{
  
  //check to ensure that the socket variable is present i.e. the browser support tests passed
  if (globalIsWebSocketOpen) 
  {
    //get the message text input element
    var webSocketMessage = document.getElementById("webSocketMessageInput").value;
    //var message = 
    //'{"event":"subscribe", "subscription":{"name":"trade"}, "pair":["XBT/USD"]}';

    if (webSocketMessage !== "") 
    {
      socket.send(webSocketMessage);
      outputConsoleMessage("Sent Message To Server: " + webSocketMessage);
    } 
    else 
    {
      alert("You must enter a WebSocket message to be sent!");
    }
  }
  else
  {
    alert("Open Socket Connection First!");
  }
}

//close the websocket connection
function closeWebSocket() 
{
  //check to ensure that the socket variable is present i.e. the browser support tests passed
  if (globalIsWebSocketOpen) 
  {
    socket.close();
  }
  else
  {
    alert("Open Socket Connection First!");
  }
}

//output message to console
function outputConsoleMessage(message)
{
  var consoleOutput = document.getElementById("outputConsole");

  var d = new Date();
  var h = d.getHours();
  var m = d.getMinutes();
  var s = d.getSeconds();
  var msg = h + ":" + m + ":" + s + " " + message;

  consoleOutput.innerHTML +=  msg  + "<br/>";
  consoleOutput.scrollTop = consoleOutput.scrollHeight;
}
