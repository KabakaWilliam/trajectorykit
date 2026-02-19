"""
Examples of using the trajectorykit agent package.

Run these examples with:
    python -m trajectorykit.examples
"""

from trajectorykit import dispatch


def example_basic():
    """Basic example: Simple math and time queries."""
    print("=" * 70)
    print("Example 1: Basic Queries")
    print("=" * 70)
    
    result = dispatch(
        user_input="What is the current time? What is 123 + 456?",
        turn_length=5,
        verbose=True
    )
    
    print("\n📊 Results:")
    print(f"  Turns: {result['turns']}")
    print(f"  Tool calls: {result['tool_calls']}")
    print(f"  Final response: {result['final_response']}")

    # Print the trace tree
    print("\n📜 Trace:")
    result["trace"].pretty_print()


def example_code_generation():
    """Example: Code generation task."""
    print("\n" + "=" * 70)
    print("Example 2: Sum Numbers")
    print("=" * 70)
    
    result = dispatch(
        user_input="Write a Python program to calculate the sum of all numbers from 1 to 100. Execute and provide the answer.",
        turn_length=5,
        verbose=True,
        max_tokens=2000
    )
    
    print("\n📊 Results:")
    print(f"  Turns: {result['turns']}")
    print(f"  Tool calls: {result['tool_calls']}")

    result["trace"].pretty_print()
    path = result["trace"].save()
    print(f"\n💾 Trace saved to: {path}")


def example_web_search():
    """Example: Web search task."""
    print("\n" + "=" * 70)
    print("Example 3: Web Search")
    print("=" * 70)
    
    result = dispatch(
        user_input="What are the latest developments in AI? Find the top 3 news items.",
        turn_length=5,
        verbose=True,
        max_tokens=3000
    )
    
    print("\n📊 Results:")
    print(f"  Turns: {result['turns']}")
    print(f"  Tool calls: {result['tool_calls']}")
    print(f"  Final response: {result['final_response'][:300]}...")

    result["trace"].pretty_print()
    # Save trace to disk
    path = result["trace"].save()
    print(f"\n💾 Trace saved to: {path}")


def example_spawn_agent():
    """Example: Recursive sub-agent delegation."""
    print("\n" + "=" * 70)
    print("Example 4: Spawn Agent (Recursive)")
    print("=" * 70)

    result = dispatch(
        user_input=(
            "I need two things done:\n"
            "1. Spawn a sub-agent to research what the Collatz conjecture is and test it on the number 27\n"
            "2. After getting the sub-agent's answer, tell me the current time"
        ),
        turn_length=5,
        verbose=True,
        max_tokens=4096
    )

    print("\n📊 Results:")
    print(f"  Turns: {result['turns']}")
    print(f"  Tool calls: {result['tool_calls']}")
    print(f"  Final response: {result['final_response']}")

    # Full trace tree — shows sub-agent internals
    print("\n📜 Full Trace Tree:")
    result["trace"].pretty_print()

    # Save trace to disk for later audit
    path = result["trace"].save()
    print(f"\n💾 Trace saved to: {path}")


def example_multi_agent():
    """Example: Multiple sub-agents working on independent subtasks."""
    print("\n" + "=" * 70)
    print("Example 5: Multi-Agent Decomposition")
    print("=" * 70)

    # result = dispatch(
    #     user_input=(
    #         "I need a comparison report. Use sub-agents to handle each part independently:\n"
    #         "1. Spawn a sub-agent to find out what the population of Tokyo is and calculate how many times larger it is than Zurich (~400,000)\n"
    #         "2. Spawn another sub-agent to write a Python program that generates the first 20 Fibonacci numbers\n"
    #         "Combine both results into a brief summary."
    #     ),
    #     turn_length=7,
    #     verbose=True,
    #     max_tokens=4096
    # )
    result = dispatch(
        user_input=(
            "Create a visualisation that compares the stats for the Blue Eyes White Dragon and The Dark Magician Yu-Gi-Oh! cards? "
            # "Compare the GDP of Japan and Germany over the last 5 years, then write a Python program to visualize the trend as a bar chart. Summarize which economy grew faster."
            # "How has the market price of the card 'Blue-Eyes White Dragon' moved over the last years? Create a line plot."
        ),
        turn_length=None,
        verbose=True,
        max_tokens=32768
    )

    print("\n📊 Results:")
    print(f"  Turns: {result['turns']}")
    print(f"  Tool calls: {result['tool_calls']}")
    print(f"  Final response: {result['final_response']}")

    # Full trace tree — shows both sub-agents and their internal tool calls
    print("\n📜 Full Trace Tree:")
    result["trace"].pretty_print()

    # Save trace to disk
    path = result["trace"].save()
    print(f"\n💾 Trace saved to: {path}")


if __name__ == "__main__":
    # example_basic()
    # example_code_generation()
    # example_web_search()
    # example_spawn_agent()
    example_multi_agent()
