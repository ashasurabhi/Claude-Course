# agent_loop.py
import json
import anthropic
from dotenv import load_dotenv


# Load the API key from the .env file next to this script
load_dotenv()

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
MODEL = "claude-haiku-5-5"
MAX_STEPS = 6

# ---- Fake data -----------------------------------------------------------

CUSTOMERS = {
    "priya@example.com": {"name": "Priya Shah", "orders": ["1001", "1002"]},
    "tom@example.com": {"name": "Tom Reid", "orders": ["1003", "1004"]},
    "lena@example.com": {"name": "Lena Park", "orders": []},
}

ORDERS = {
    "1001": {
        "customer": "priya@example.com",
        "placed": "2026-10-20",
        "status": "shipped",
        "eta": "2026-10-28",
        "items": [{"sku": "KB-100", "name": "Keyboard", "qty": 1}],
    },
    "1002": {
        "customer": "priya@example.com",
        "placed": "2026-10-24",
        "status": "processing",
        "eta": "2026-11-02",
        "items": [
            {"sku": "MS-220", "name": "Mouse", "qty": 1},
            {"sku": "CB-05", "name": "USB cable", "qty": 2},
        ],
    },
    "1003": {
        "customer": "tom@example.com",
        "placed": "2026-10-15",
        "status": "delivered",
        "eta": "2026-10-22",
        "items": [{"sku": "MN-27", "name": "Monitor", "qty": 1}],
    },
    "1004": {
        "customer": "tom@example.com",
        "placed": "2026-10-25",
        "status": "processing",
        "eta": "2026-11-05",
        "items": [{"sku": "HD-9", "name": "Headset", "qty": 1}],
    },
}

# ---- Tools ---------------------------------------------------------------

def find_customer_orders(email: str) -> dict:
    if email not in CUSTOMERS:
        raise ValueError(f"No customer with email {email}")
    customer = CUSTOMERS[email]
    return {"name": customer["name"], "order_ids": customer["orders"]}

def get_order_status(order_id: str) -> dict:
    if order_id not in ORDERS:
        raise ValueError(f"No order with id {order_id}")
    order = ORDERS[order_id]
    return {"order_id": order_id, "status": order["status"], "eta": order["eta"]}

def get_order_items(order_id: str) -> dict:
    if order_id not in ORDERS:
        raise ValueError(f"No order with id {order_id}")
    return {"order_id": order_id, "items": ORDERS[order_id]["items"]}

TOOLS = [
    {
        "name": "find_customer_orders",
        "description": "Return a customer's name and order numbers, given their email address. Use this first when the customer doesn't give an order number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "The customer's email address"}
            },
            "required": ["email"],
        },
    },
    {
        "name": "get_order_status",
        "description": "Return the shipping status and estimated delivery date of one order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order number, for example 1001"}
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "get_order_items",
        "description": "Return the items in one order. Use this when the customer asks what they bought.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order number, for example 1001"}
            },
            "required": ["order_id"],
        },
    },
]

TOOL_FUNCTIONS = {
    "find_customer_orders": find_customer_orders,
    "get_order_status": get_order_status,
    "get_order_items": get_order_items,
}

# ---- Agent loop ----------------------------------------------------------

def run_tool(block) -> dict:
    """Run one tool request and return a tool_result. Errors go back to Claude as results."""
    print(f"    -> {block.name}({block.input})")
    try:
        output = TOOL_FUNCTIONS[block.name](**block.input)
        return {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(output)}
    except Exception as e:
        print(f"       error: {e}")
        return {"type": "tool_result", "tool_use_id": block.id, "content": str(e), "is_error": True}

def agent(question: str) -> str:
    messages = [{"role": "user", "content": question}]

    for _ in range(MAX_STEPS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "max_tokens":
            return "Stopped: the answer was cut off. Try a shorter question."

        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        results = [run_tool(block) for block in response.content if block.type == "tool_use"]
        messages.append({"role": "user", "content": results})

    return "Stopped: reached the step limit"

# ---- Run it --------------------------------------------------------------

if __name__ == "__main__":
    questions = [
        "I'm priya@example.com. When is my latest order arriving?",
        "Tom here, tom@example.com. What did I buy in order 1003?",
        "Is order 9999 shipped yet?",
        "I'm lena@example.com. Do I have any orders?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {agent(q)}")