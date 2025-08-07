"""
OpenAI Response API with MCP VectorDB Integration

This example uses OpenAI's new Response API that can directly call
external tools (MCP VectorDB) without manual function handling.
"""

import asyncio
import json
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from dotenv import dotenv_values
import openai

# Load environment
env_vars = dotenv_values("./.env")
client = openai.OpenAI(api_key=env_vars.get("OPENAI_API_KEY"))

# MCP server setup
server_params = StdioServerParameters(
    command="python",
    args=["-m", "main"],
    env=env_vars,
)

async def convert_mcp_tools_to_openai_format(tools_list):
    """Convert MCP tools to OpenAI Response API format"""
    openai_tools = []
    
    for tool in tools_list.tools:
        # Extract tool information
        tool_name = tool.name
        tool_description = tool.description or f"Execute {tool_name} tool"
        
        # Convert input schema to Response API format
        openai_tool = {
            "type": "function", # there are different types and if we pass mcp, it automatically call the mcp server and invoke it.
            "name": tool_name,
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        
        # Process input schema if available
        if hasattr(tool, 'inputSchema') and tool.inputSchema:
            schema = tool.inputSchema
            if isinstance(schema, dict):
                # Copy properties
                if "properties" in schema:
                    openai_tool["parameters"]["properties"] = schema["properties"]
                
                # Copy required fields
                if "required" in schema:
                    openai_tool["parameters"]["required"] = schema["required"]
        
        openai_tools.append(openai_tool)
    
    return openai_tools

# I need to invoke the tool, as I do not have the MCP server hosted 
async def create_mcp_response_handler(session):
    """Create a response handler that integrates with MCP"""
    
    async def mcp_tool_handler(tool_name: str, arguments: dict):
        """Handle MCP tool calls"""
        print(f"🔧 Executing MCP tool: {tool_name} with {arguments}")
        
        try:
            result = await session.call_tool(name=tool_name, arguments=arguments)
            return result.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"
    
    return mcp_tool_handler

async def main():
    async with stdio_client(server=server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            # Get available MCP tools
            tools_list = await session.list_tools()
            print(f"📋 Available MCP tools: {[tool.name for tool in tools_list.tools]}")
            
            # Create MCP tool handler
            mcp_handler = await create_mcp_response_handler(session)
            
            # Convert MCP tools to OpenAI format dynamically
            tools = await convert_mcp_tools_to_openai_format(tools_list)
            print(f"🔧 Converted {len(tools)} tools for OpenAI API")
            
            # User request
            user_input = """
            Store these facts:
            1. "OpenAI released GPT-4 in 2023"
            2. "Vector databases are used for similarity search"
            
            """
            
            print(f"👤 User: {user_input}")
            print("\n" + "="*60)
            
            # Use OpenAI Response API
            response = client.responses.create(
                model="o4-mini",
                input=user_input,
                instructions="You help users store the documents. Use the tools to complete their requests.",
                tools=tools,
                tool_choice="auto",
                max_tool_calls=2,
            )

            
            # Process the response - Response API has different structure
            print(f"🤖 Response received: {type(response)}")
            print(f"📋 Response status: {response.status}")
            
            # Process multiple rounds of tool calls
            conversation_history = []
            current_response = response
            
            for round_num in range(5):  # Max 5 rounds to prevent infinite loops
                if hasattr(current_response, 'output') and current_response.output:
                    tool_calls = [item for item in current_response.output if hasattr(item, 'type') and item.type == 'function_call']
                    
                    if tool_calls:
                        print(f"🤖 Round {round_num + 1}: OpenAI wants to call {len(tool_calls)} tools:")
                        
                        # Execute each tool call through MCP
                        for tool_call in tool_calls:
                            func_name = tool_call.name
                            args = json.loads(tool_call.arguments)
                            
                            print(f"\n📞 Tool: {func_name}")
                            print(f"📝 Args: {args}")
                            
                            # Call MCP tool
                            result = await mcp_handler(func_name, args)
                            print(f"✅ Result: {result[:150]}...")
                            
                            # Add to conversation history
                            conversation_history.append(f"Executed {func_name} with result: {result}")
                        
                        # Continue conversation to see if more tools are needed
                        next_input = f"Previous actions completed: {'; '.join(conversation_history)}. Continue with the original request if more actions are needed."
                        
                        current_response = client.responses.create(
                            model="gpt-4",
                            input=next_input,
                            instructions="You help users store and search documents. Use the tools to complete their requests step by step. If you have completed all requested actions, provide a summary.",
                            tools=tools,
                            tool_choice="auto"
                        )
                        
                        print(f"📋 Round {round_num + 1} completed")
                    else:
                        # No more tool calls, check for final response
                        if hasattr(current_response, 'output_text'):
                            print(f"💬 Final Response: {current_response.output_text}")
                        else:
                            print(f"💬 Final Response: {current_response}")
                        break
                else:
                    # No output, end conversation
                    if hasattr(current_response, 'output_text'):
                        print(f"💬 Final Response: {current_response.output_text}")
                    else:
                        print(f"💬 Final Response: {current_response}")
                    break
            
            print(f"\n🎯 Conversation completed after {round_num + 1} rounds")
            print(f"📝 Actions performed: {len(conversation_history)}")
            for i, action in enumerate(conversation_history, 1):
                print(f"   {i}. {action[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
