import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()
async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
    )

    try:
        analyst = AssistantAgent(
            name="analyst",
            model_client=model_client,
            system_message=(
                "Analyze the given information. "
                "Return the key facts clearly."
            ),
        )

        summarizer = AssistantAgent(
            name="summarizer",
            model_client=model_client,
            system_message=(
                "Summarize the information you receive. "
                "Do not invent missing facts."
            ),
        )

        task = (
            "A store sold 120 products on Monday and 150 on Tuesday. "
            "Revenue was $600 on Monday and $750 on Tuesday."
        )
        analysis_result = await analyst.run(task=task)
        analysis_text = analysis_result.messages[-1].content

        print("\n--- ANALYST OUTPUT ---")
        print(analysis_text)

        summary_result = await summarizer.run(
            task=(
                "Summarize these findings:\n\n"
                + analysis_text
            )
        )

        print("\n--- SUMMARIZER OUTPUT ---")
        print(summary_result.messages[-1].content)

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
