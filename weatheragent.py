from dotenv import load_dotenv
import anthropic

# Load the API key from the .env file next to this script
load_dotenv()

client = anthropic.Anthropic()

# The tools array tells Claude what's available:
# a name, a description, and a JSON schema for the inputs.
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get weather for",
                }
            },
            "required": ["city"],
        },
    }
]


# run_tool is just a hardcoded lookup.
# In a real app, this would hit your database, an API, whatever.
def run_tool(name, tool_input):
    if name == "get_weather":
        return f"Weather in {tool_input['city']}: 95F, sunny"
    raise ValueError(f"Unknown tool: {name}")


messages = [
    {"role": "user", "content": "What should I wear in Austin today?"}
]

# Safety limit so the loop can never run forever
MAX_ROUNDS = 10

for round_number in range(1, MAX_ROUNDS + 1):
    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )

    # Shows what happened in each round. Remove once you understand the flow.
    print(f"\n--- Round {round_number}: stop_reason = {response.stop_reason} ---")

    if response.stop_reason == "end_turn":
        # Claude is done. Print the final text and stop.
        for block in response.content:
            if block.type == "text":
                print(block.text)
        break

    elif response.stop_reason == "tool_use":
        # Find the tool use blocks in the response and run each one.
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"Claude called {block.name} with {block.input}")
                result = run_tool(block.name, block.input)
                print(f"Result: {result}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        # Send Claude's response and our tool results back,
        # then loop again so Claude can answer.
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    else:
        # Anything else (for example max_tokens) would otherwise loop forever
        print(f"Stopped unexpectedly: {response.stop_reason}")
        break

else:
    print(f"Gave up after {MAX_ROUNDS} rounds without a final answer.")