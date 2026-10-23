
import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


def addition(a: int, b: int) -> int:
    """Add two integers."""
    if type(a) is not int or type(b) is not int:
        raise ValueError("Inputs must be integers.")
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
        calculator = AssistantAgent(
            name="calculator",
            model_client=model_client,
            tools=[addition],
            max_tool_iterations=3,
            system_message=(
                "You are a calculator. "
                "Use the addition tool to calculate the answer. "
                "Report the result clearly."
            ),
        )

        reviewer = AssistantAgent(
            name="reviewer",
            model_client=model_client,
            system_message=(
                "You are a calculation reviewer. "
                "Check the previous agent's arithmetic result. "
                "If it is correct, explain briefly and end your response "
                "with the exact word APPROVED. "
                "If it is incorrect, explain the issue."
            ),
        )
        termination = TextMentionTermination("APPROVED")
        team = SelectorGroupChat(
            participants=[calculator, reviewer],
            model_client=model_client,
            max_turns=4,
            termination_condition=termination,
        )

        result = await team.run(
            task="Calculate 10 + 20 and review the result."
        )

        for message in result.messages:
            print("\n" + "=" * 40)
            print("SOURCE:", message.source)
            print("TYPE:", type(message).__name__)
            print("CONTENT:", repr(message.content))

        print("\nSTOP REASON:", result.stop_reason)

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
