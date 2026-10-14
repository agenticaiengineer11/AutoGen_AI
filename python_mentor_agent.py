
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

import os
import asyncio


async def main():
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Check your .env file."
        )

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=openai_api_key,
    )

    try:
        assistant = AssistantAgent(
            name="python_mentor",
            model_client=model_client,
            system_message="""
            You are a professional Python programming mentor.
            Explain concepts clearly and provide beginner-friendly
            guidance with simple examples.
            """,
            verbose= True
        )

        result = await assistant.run(
            task=(
                "I want to learn Python programming. "
                "Can you help me? Give me a simple answer."
            )
        )

        for message in result.messages:
            print(f"\n[{message.source}]")
            print(message.content)

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
