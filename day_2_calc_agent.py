
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

import os
import asyncio


def multiply(a: int, b: int) -> int:
    """Multiply two integers. Booleans are not allowed."""

    if type(a) is not int:
        raise ValueError("a must be an integer, not a boolean.")

    if type(b) is not int:
        raise ValueError("b must be an integer, not a boolean.")

    return a * b


def addition(a: int, b: int) -> int:
    """Add two integers. Booleans are not allowed."""

    if type(a) is not int:
        raise ValueError("a must be an integer, not a boolean.")

    if type(b) is not int:
        raise ValueError("b must be an integer, not a boolean.")

    return a + b


async def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing.")

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key,
    )

    try:
        calculator_agent = AssistantAgent(
            name="calculator_agent",
            model_client=model_client,
            tools=[addition, multiply],
            max_tool_iterations = 5,
            system_message=(
                "You are a calculator agent. "
                "Use the provided tools for calculations. "
                "For multi-step calculations, use the result "
                "of one operation as the input to the next. "
                "For the user's requested arithmetic, use the tools "
                "rather than calculating mentally."
            ),
        )

        result = await calculator_agent.run(
            task="Add 10 and 20, then multiply the result by 3."
        )

        for message in result.messages:
            print("\n" + "=" * 40)
            print("SOURCE:", message.source)
            print("TYPE:", type(message).__name__)
            print("CONTENT:", repr(message.content))

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
