# agent_toolrunner.py
import json
import anthropic
from anthropic import beta_tool
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
# @beta_tool builds each tool's schema from the type hints and docstring.
# If a tool raises, the tool runner sends the error back to Claude as an
# is_error tool_result, so the tools can simply raise.

@beta_tool
def find_customer_orders(email: str) -> str:
    """Return a customer's name and order numbers, given their email address. Use this first when the customer doesn't give an order number.

    Args:
        email: The customer's email address
    """
    print(f"    -> find_customer_orders({email!r})")
    if email not in CUSTOMERS:
        raise ValueError(f"No customer with email {email}")
    customer = CUSTOMERS[email]
    return json.dumps({"name": customer["name"], "order_ids": customer["orders"]})

@beta_tool
def get_order_status(order_id: str) -> str:
    """Return the shipping status and estimated delivery date of one order.

    Args:
        order_id: The order number, for example 1001
    """
    print(f"    -> get_order_status({order_id!r})")
    if order_id not in ORDERS:
        raise ValueError(f"No order with id {order_id}")
    order = ORDERS[order_id]
    return json.dumps({"order_id": order_id, "status": order["status"], "eta": order["eta"]})

@beta_tool
def get_order_items(order_id: str) -> str:
    """Return the items in one order. Use this when the customer asks what they bought.

    Args:
        order_id: The order number, for example 1001
    """
    print(f"    -> get_order_items({order_id!r})")
    if order_id not in ORDERS:
        raise ValueError(f"No order with id {order_id}")
    return json.dumps({"order_id": order_id, "items": ORDERS[order_id]["items"]})

TOOLS = [find_customer_orders, get_order_status, get_order_items]

# ---- Agent ---------------------------------------------------------------

def agent(question: str) -> str:
    # The tool runner owns the loop: call Claude, run requested tools, send results back.
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=1024,
        tools=TOOLS,
        messages=[{"role": "user", "content": question}],
        max_iterations=MAX_STEPS,
    )
    response = runner.until_done()

    if response.stop_reason == "max_tokens":
        return "Stopped: the answer was cut off. Try a shorter question."
    if response.stop_reason == "tool_use":
        return "Stopped: reached the step limit"
    return "".join(block.text for block in response.content if block.type == "text")

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