from mcp import ClientSession
from mcp.client.sse import sse_client
import asyncio
import logging

logger = logging.getLogger('mcp_client')

async def main():
   
    async with sse_client(f"http://localhost:8000/sse") as(
        read_stream,
        write_stream
    ):
        async with ClientSession(
            read_stream,
            write_stream
        ) as session:
            await session.initialize()
            print("Session initialized, ready to call tools.")
            print("\n")
           
            tool_list = await session.list_tools()
            print(f"Tool result: {tool_list}")  
            print("\n")
            
            res = await session.call_tool(name="store_text", arguments={"text":"hey my name is gowtham", "collection":"memory", "metadata" :{"user":"gowtham"}}, ) 
            print(f"Tool result: {res}")    
            print("\n")
            
            search_res = await session.call_tool(name="similarity_search", arguments={"query":"what is my name", "collection":"memory", "top_k":10})    
            print(f"Tool result: {search_res}")    
            print("\n")
            



if __name__ == "__main__":
    # Classic HTTP streaming client mode
    asyncio.run(main())


# SSE - Server Sent Events
'''
SSE - Server Sent Events

SSE (Server-Sent Events) is a one-way, server-to-client streaming protocol over HTTP.

Client connect to the server and get the session ID by /sse and  server will return the all the responses in this channel
And client request the server with the session ID HTTP POST /messages for the request in different channel.
Two channels:
 stream channel (/sse)
 request channel (/message?sessionId=)
The response does NOT come in the POST response
It arrives on the SSE stream.

 Flow:

 - Server is initialized
 - Client create the connection "GET /sse HTTP/1.1" - it is automatically done by the client before  await session.initialize()
 - /sse endpoint is opened—this stream will deliver updates asynchronously from server to client.
 -  await session.initialize() is for us to send request
 - Client send request via "POST /messages/?session_id=c318341d54874c788c9e74a37a9e9c8a HTTP/1.1"
 - Server processes request
 - Server pushes results back to the SSE stream, as they become available.

 SSE is one direction, But that doesn’t mean the client can’t talk to the server — it just does so through standard HTTP requests, separately from the SSE stream.
 Client sends requests using POST /messages?...
 Server pushes results back using the persistent SSE stream (GET /sse)
---------------------------------------------

MCP moved to streamable HTTP from SSE due to the following reasons,
 - Stremable HTTP is bidirection and maintain the single session for full communication

Client sends request
Server returns response in the same HTTP response
SSE only used if server wants to push async events



'''
