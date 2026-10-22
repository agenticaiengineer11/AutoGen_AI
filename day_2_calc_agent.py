from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import asyncio
import os 

load_dotenv()
def multiply(a: int, b:int )-> int:
    """Multiplies two integers and returns the result."""
    return a * b

async def main():
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is missing.Check your .env file.")

    model_client = OpenAIChatCompletionClient(
        model = "gpt-4o-mini",
        api_key = openai_api_key
    )

    try:
        assistant = AssistantAgent(
            name = "Calculator Assistant",
            model_client = model_client,
            tools = [multiply],
            system_message = "You are a helpul AI assistant that can perform basic arithmetic operations like addition, subtraction, multiplication, and division."
        )

        result = await assistant.run(
            task = "Multiply 16 and 17 and provide the result."
        )

    finally:
        await model_client.close()