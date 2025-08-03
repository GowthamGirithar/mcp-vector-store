from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import asyncio
import logging

logger = logging.getLogger('mcp_client')

async def main():
    # Streamable HTTP client mode with bidirectional connection
    # read_stream to receive the response
    # write_stream to send the request
    # async with - is async version and with is resoure managenemtn
    # as -> return streams ans assigned to the variable
    async with streamablehttp_client(f"http://localhost:8000/mcp") as(
        read_stream,
        write_stream,
        session_callback,
    ):
        async with ClientSession(
            read_stream,
            write_stream
        ) as session:
            id_before = session_callback()
            print(f"Session ID before init: {id_before}")
            await session.initialize()
            id_after = session_callback()
            print(f"Session ID after init: {id_after}")
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


# Server started with transport streamable-http
# Client create the session and session ID created - "POST /mcp HTTP/1.1" 202 Accepted - Session Start
# Lifecycle hook executed
# Client connects to /mcp, session ID assigned
# "GET /mcp HTTP/1.1" 200 OK - client listening for responses or events from the server. 
    # - Open once, keep connection open; Since it is bidirectionals, POST is responsbile for sending the request and GET which listening to connection responsible for listeing to response
# ListTools  POST /mcp & CallTool POST /mcp requests are processed
# Client shutdown - Session ends DELETE /mcp client terminate the session
# Lifecycle hook executed - cleanup
# server stopped
# when agent initiated, it create the connection and maintain it for long time and so only we have lifecycle hook as session lifecycle hook