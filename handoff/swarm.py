import asyncio
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import TextMentionTermination
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
load_dotenv()
async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini"
    )

    try:
        triage_agent  = AssistantAgent(
            name=  "triage_agent",
            model_client=model_client,
            handoffs=["research_agent"],
            system_message="""
            You are a triage agent.
            Immediately hand off every user request to research_agent.
            Do not answer the user's request yourself.
            """,
        )

        research_agent = AssistantAgent(
            model_client=model_client,
            name = "research_agent",
            system_message="""
            You are a research specialist.
            Answer the user's request clearly and accurately.
            After completing the answer, include the exact word DONE.
            """,
        )

        team = Swarm(
            participants=[triage_agent, research_agent],
            max_turns=5,
            termination_condition=TextMentionTermination("DONE"),


        )
        result = await team.run(
            task="Explain why multi-agent systems use handoffs, "
                 "and give one practical example."
        )

        for message in result.messages:
            print("\n" + "=" * 45)
            print("SOURCE:", message.source)
            print("TYPE:", type(message).__name__)
            print("CONTENT:", message.content)

        print("\nSTOP REASON:", result.stop_reason)

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
