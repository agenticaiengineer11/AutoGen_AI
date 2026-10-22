from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

import os
import asyncio


def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b

def addition(a:int , b:int)-> int:
    """Add two integers and return the result."""
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
            task="What is 125 plus 48?"
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