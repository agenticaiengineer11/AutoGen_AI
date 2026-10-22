from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

import os
import asyncio


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    if not isinstance(a, int) or isinstance(a, bool):
        raise ValueError("a must be an integer.")

    if not isinstance(b, int) or isinstance(b, bool):
        raise ValueError("b must be an integer.")

    return a * b

def addition(a:int , b:int)-> int:
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
        assistant = AssistantAgent(
            name="calculator_agent",
            model_client=model_client,
            tools=[multiply, addition],
            system_message=(
                "You are a calculator assistant. "
                "Use the multiply tool for multiplication questions and the addition tool for addition questions. "
                "Explain the result clearly."
            ),
        )

        result = await assistant.run(
            task="Use the add tool to calculate True + 48"
        )

        for message in result.messages:
            print("\n" + "=" * 40)
            print("SOURCE:", message.source)
            print("TYPE:", type(message).__name__)
            print("CONTENT:", message.content)

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())