import asyncio
import sys

import anthropic
from dotenv import load_dotenv
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
client = anthropic.AsyncAnthropic()
MODEL = "claude-haiku-5-5"

NPX = "npx.cmd" if sys.platform == "win32" else "npx"

# Read-only: the server can't create, change or delete model objects
SERVER_ARGS = ["-y", "@microsoft/powerbi-modeling-mcp@latest", "--start", "--readonly"]


async def ask(question: str) -> str:
    params = StdioServerParameters(command=NPX, args=SERVER_ARGS)

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            listed = await session.list_tools()
            print("Server offers:", [t.name for t in listed.tools])

            runner = client.beta.messages.tool_runner(
                model=MODEL,
                max_tokens=2048,
                tools=[async_mcp_tool(t, session) for t in listed.tools],
                messages=[{"role": "user", "content": question}],
                max_iterations=8,
            )

            # The runner calls the MCP tools and loops; we just watch each assistant turn.
            response = None
            async for response in runner:
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  -> {block.name}({block.input})")

    if response is None or response.stop_reason == "tool_use":
        return "Stopped: reached the step limit"
    return "".join(b.text for b in response.content if b.type == "text")


if __name__ == "__main__":
    question = (
        r"Connect to the Power BI project folder at C:\Claude\Claude-Course\model-copy\Campaign Analysis.SemanticModel\definition"
        "(a copy, not the original). List the tables and measures. "
            )
    print(asyncio.run(ask(question)))