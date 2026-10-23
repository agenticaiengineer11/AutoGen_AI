
import asyncio
from autogen_agentchat.agents import UserProxyAgent


async def main():
    user_proxy = UserProxyAgent(
        name="human",
        input_func=input,
    )

    result = await user_proxy.run(
        task="Should the proposed action be approved? Reply with yes or no."
    )

    for message in result.messages:
        print(message.source, message.content)


if __name__ == "__main__":
    asyncio.run(main())
