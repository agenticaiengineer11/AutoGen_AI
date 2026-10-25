import asyncio
import json
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

def create_team(model_client):
    researcher = AssistantAgent(
        name="researcher",
        model_client=model_client,
        system_message=(
            "You are a researcher. Summarize facts clearly. "
            "Do not invent information."
        ),
    )

    summarizer = AssistantAgent(
        name="summarizer",
        model_client=model_client,
        system_message=(
            "You summarize the conversation. "
            "Use the available conversation history. "
            "Do not invent facts."
        ),
    )

    return RoundRobinGroupChat(
        participants=[researcher, summarizer],
        termination_condition=MaxMessageTermination(max_messages=3),
    )


async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini"
    )

    try:
        team1 = create_team(model_client)

        result1 = await team1.run(
            task=(
                "Remember these project facts: "
                "Project name is Orion. "
                "Project budget is $12,000. "
                "The deadline is December 15, 2026. "
                "Summarize these facts."
            )
        )

        print("\n--- ORIGINAL TEAM OUTPUT ---")
        for message in result1.messages:
            print(f"{message.source}: {message.content}")
        state = await team1.save_state()

        with open("team_state.json", "w", encoding="utf-8") as file:
            json.dump(state, file, indent=2)

        print("\nTeam state saved successfully.")
        team2 = create_team(model_client)
        with open("team_state.json", "r", encoding="utf-8") as file:
            saved_state = json.load(file)

        await team2.load_state(saved_state)

        print("Team state restored successfully.")
        result2 = await team2.run(
            task=(
                "From the earlier project conversation, "
                "what are the project name, budget, and deadline? "
                "Answer using the saved conversation context."
            )
        )

        print("\n--- RESTORED TEAM OUTPUT ---")
        for message in result2.messages:
            print(f"{message.source}: {message.content}")

    finally:
        await model_client.close()


asyncio.run(main())
