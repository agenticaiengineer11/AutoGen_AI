
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()


async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini"
    )

    try:
        assistant = AssistantAgent(
            name="sales_assistant",
            model_client=model_client,
            system_message=(
                "You are a sales assistant. "
                "Use the information provided in the conversation. "
                "Do not invent missing sales figures."
            ),
        )

        # First task: provide context
        result1 = await assistant.run(
            task=(
                "Remember this information for this example: "
                "Monday sales were 120 units and revenue was $600. "
                "Summarize these facts."
            )
        )

        print("\n--- FIRST RESPONSE ---")
        print(result1.messages[-1].content)

        # Second task: explicitly provide the earlier facts again
        result2 = await assistant.run(
            task=(
                "Using these facts: Monday sales were 120 units "
                "and revenue was $600. "
                "What was Monday's revenue per unit?"
            )
        )

        print("\n--- SECOND RESPONSE ---")
        print(result2.messages[-1].content)

    finally:
        await model_client.close()


asyncio.run(main())
