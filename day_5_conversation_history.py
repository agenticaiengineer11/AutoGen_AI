import asyncio
import json

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()


async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini"
    )

    try:
        agent1 = AssistantAgent(
            name="sales_assistant",
            model_client=model_client,
            system_message=(
                "You are a sales assistant. "
                "Use facts available in your conversation. "
                "Do not invent missing information."
            ),
        )
        result1 = await agent1.run(
            task=(
                "Our project code is ALPHA-27. "
                "Monday sales were 120 units. "
                "Confirm these facts briefly."
            )
        )

        print("\n--- FIRST RESPONSE ---")
        print(result1.messages[-1].content)

        saved_state = await agent1.save_state()

        with open("sales_agent_state.json", "w", encoding="utf-8") as file:
            json.dump(saved_state, file, indent=2)

        print("\nAgent state saved successfully.")
        agent2 = AssistantAgent(
            name="sales_assistant",
            model_client=model_client,
            system_message=(
                "You are a sales assistant. "
                "Answer using your available conversation context. "
                "Do not invent missing information."
            ),
        )
        with open("sales_agent_state.json", "r", encoding="utf-8") as file:
            loaded_state = json.load(file)

        await agent2.load_state(loaded_state)

        result2 = await agent2.run(
            task="What was our project code and Monday's sales?"
        )

        print("\n--- RESPONSE AFTER RESTORING STATE ---")
        print(result2.messages[-1].content)

    finally:
        await model_client.close()


asyncio.run(main())
