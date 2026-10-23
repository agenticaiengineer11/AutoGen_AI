import asyncio
from autogen_agentchat.agents import UserProxyAgent


def perform_action(action: str) -> str:
    """Execute an action after approval."""
    return f"Action executed: {action}"


async def main():
    user_proxy = UserProxyAgent(
        name="human",
        input_func=input,
    )

    action = "generate a project report"

    print(f"Proposed action: {action}")
    approval = await user_proxy.run(
        task="Approve this action? Type exactly yes or no: "
             + action
    )

    response = approval.messages[-1].content.strip().lower()

    if response == "yes":
        print(perform_action(action))
    else:
        print("Action cancelled: approval not granted.")


if __name__ == "__main__":
    asyncio.run(main())