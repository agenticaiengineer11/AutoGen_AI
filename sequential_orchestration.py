import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
    )

    try:
        research_agent = AssistantAgent(
            name="research_agent",
            model_client=model_client,
            system_message=(
                "Explain compound interest in simple terms. "
                "Be concise and accurate."
            ),
        )

        math_agent = AssistantAgent(
            name="math_agent",
            model_client=model_client,
            system_message=(
                "Calculate compound interest accurately. "
                "Assume annual compounding unless specified. "
                "Show the formula and final amount."
            ),
        )

        reviewer_agent = AssistantAgent(
            name="reviewer_agent",
            model_client=model_client,
            system_message=(
                "Review the explanation and calculation provided. "
                "Check whether they are consistent and correct. "
                "Point out any errors you find."
            ),
        )

        task = (
            "Explain compound interest and calculate the final "
            "amount for $1,000 at 5% annual interest over 3 years."
        )

        research_result = await research_agent.run(task=task)
        research_text = research_result.messages[-1].content

        print("\n--- RESEARCH AGENT ---")
        print(research_text)

        math_result = await math_agent.run(task=task)
        math_text = math_result.messages[-1].content

        print("\n--- MATH AGENT ---")
        print(math_text)

        review_task = (
            f"Original task: {task}\n\n"
            f"Research agent output:\n{research_text}\n\n"
            f"Math agent output:\n{math_text}\n\n"
            "Review both outputs for correctness and consistency."
        )

        review_result = await reviewer_agent.run(task=review_task)

        print("\n--- REVIEWER AGENT ---")
        print(review_result.messages[-1].content)

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
