from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
import asyncio
import logging
from dotenv import dotenv_values

logger = logging.getLogger('mcp_client')

env_vars = dotenv_values("./.env") 
stdioParam = StdioServerParameters(
            command="python",  # Executable
            args=["-m", "main"],  # Server script
            env=env_vars,  # Optional environment variables
        )

async def main():
   
    async with stdio_client(server=stdioParam) as(
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




