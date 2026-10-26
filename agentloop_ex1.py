import json
import anthropic
from dotenv import load_dotenv


# Load the API key from the .env file next to this script
load_dotenv()
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
MODEL = "claude-haiku-5-5"

# Fake order database. In real life this would be a database query or an API call.
ORDERS = {
    "1001": {"status": "shipped", "eta": "2026-10-28"},
    "1002": {"status": "processing", "eta": "2026-11-02"},
}

def get_order_status(order_id: str) -> dict:
    if order_id not in ORDERS:
        raise ValueError(f"No order with id {order_id}")
    return ORDERS[order_id]

# The tool description is what Claude reads to decide when to use it.
TOOLS = [
    {
        "name": "get_order_status",
        "description": "Look up the shipping status and estimated delivery date of a customer order. Use this whenever the customer asks about an order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order number, for example 1001"}
            },
            "required": ["order_id"],
        },
    }
]

TOOL_FUNCTIONS = {"get_order_status": get_order_status}

def run_tool(block) -> dict:
    """Run one tool call and return a tool_result block. Errors go back to Claude as results."""
    try:
        output = TOOL_FUNCTIONS[block.name](**block.input)
        return {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(output)}
    except Exception as e:
        return {"type": "tool_result", "tool_use_id": block.id, "content": str(e), "is_error": True}

def agent(question: str, max_steps: int = 5) -> str:
    messages = [{"role": "user", "content": question}]

    for _ in range(max_steps):  # cap the loop so a confused model can't run forever
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        # No tool requested means Claude has its final answer
        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        # Run each requested tool and send all the results back in one user message
        results = [run_tool(block) for block in response.content if block.type == "tool_use"]
        messages.append({"role": "user", "content": results})

    return "Stopped: reached the step limit"

if __name__ == "__main__":
    print(agent("Where is my order 1001?"))