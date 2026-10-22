from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os 

import asyncio
load_dotenv()
def addition(a: int , b:int )-> int:
    """Add two integers. Booleans are not allowed."""

    if type(a) is not int:
        raise ValueError("a must be an integer, not a boolean.")

    if type(b) is not int:
        raise ValueError("b must be an integer, not a boolean.")

    return a + b
async def main():
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=openai_api_key,
    )

    try:
        Calculator = AssistantAgent(
            name="calculator",
            model_client=model_client,
            system_message="You are a helpful AI assistant.",
        )

        Explainer = AssistantAgent(
            name = "explainer",
            model_client = model_client,
            system_message ="Explain the calculation to the beginner in simple terms."
        )

        

        calculation = await Calculator.run(
            task="Use the addition tool to add 10 and 20."
        )

        print("CALCULATOR RESULT:")
        print(calculation.messages[-1].content)

        explanation = await Explainer.run(
            task=(
                "Explain this calculation to a beginner: "
                f"{calculation.messages[-1].content}"
            )
        )

        print("EXPLANATION:")
        print(explanation.messages[-1].content)

    finally:
        await model_client.close()
if __name__ == "__main__":
    asyncio.run(main())