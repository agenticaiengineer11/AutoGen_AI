
import asyncio
from autogen_agentchat.agents import UserProxyAgent


def perform_action(action: str, approved: bool) -> str:
    """Execute an action only when approval is explicitly granted."""

    if approved is not True:
        return "BLOCKED: Human approval is required."

    return f"Action executed: {action}"


async def main():
    user_proxy = UserProxyAgent(
        name="human",
        input_func=input,
    )

    action = "generate a project report"

    print(f"Proposed action: {action}")

    approval = await user_proxy.run(
        task=f"Approve this action? Type exactly yes or no: {action}"
    )

    response = approval.messages[-1].content.strip().lower()

    if response == "yes":
        print(perform_action(action, approved=True))
    else:
        print(perform_action(action, approved=False))


if __name__ == "__main__":
    asyncio.run(main())
