
import asyncio
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import Swarm
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()


async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        parallel_tool_calls=False
    )

    try:
        research_agent = AssistantAgent(
            name="research_agent",
            model_client=model_client,
            system_message="""
            You handle general research and explanations.
            Answer the user's question clearly.
            End your final answer with DONE.
            """,
        )

        math_agent = AssistantAgent(
            name="math_agent",
            model_client=model_client,
            system_message="""
            You handle arithmetic and mathematical questions.
            Calculate carefully and explain the result.
            End your final answer with DONE.
            """,
        )

        triage_agent = AssistantAgent(
            name="triage_agent",
            model_client=model_client,
            handoffs=["research_agent", "math_agent"],
           system_message="""
                You are a triage agent responsible for routing user requests.

                Available specialists:
                - math_agent: arithmetic, calculations, formulas, numerical problems.
                - research_agent: explanations, conceptual questions, comparisons, general research.

                Routing rules:
                1. If the request requires a calculation or numerical result,
                    hand off to math_agent.
                2. If the request only asks for an explanation or research,
                    hand off to research_agent.
                3. If the request requires BOTH explanation and calculation,
                    hand off to math_agent first.
                4. Do not answer the user's question yourself.
                5. Only hand off to the configured specialist agents.
                6. Choose exactly ONE specialist per handoff.
                    Never request multiple handoffs in the same response.
                    For requests involving both explanation and calculation,
                    hand off to math_agent first.
            """,

        )

        team = Swarm(
            participants=[
                triage_agent,
                research_agent,
                math_agent,
            ],
            termination_condition=TextMentionTermination("DONE"),
            max_turns=6,
        )

        task = input("Enter your task: ")

        result = await team.run(task=task)

        for message in result.messages:
            print("\n" + "=" * 40)
            print("SOURCE:", message.source)
            print("TYPE:", type(message).__name__)
            print("CONTENT:", message.content)

        print("\nSTOP REASON:", result.stop_reason)

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
