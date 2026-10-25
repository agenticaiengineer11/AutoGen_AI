import asyncio
import json
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

def validate_sales_data(data: dict) -> dict:
    """Validate sales fields and calculate sales metrics."""

    required_fields = [
        "monday_units",
        "tuesday_units",
        "monday_revenue",
        "tuesday_revenue",
    ]

    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    monday_units = data["monday_units"]
    tuesday_units = data["tuesday_units"]
    monday_revenue = data["monday_revenue"]
    tuesday_revenue = data["tuesday_revenue"]
    if type(monday_units) is not int or monday_units <= 0:
        raise ValueError("monday_units must be a positive integer.")

    if type(tuesday_units) is not int or tuesday_units <= 0:
        raise ValueError("tuesday_units must be a positive integer.")

    for field, value in [
        ("monday_revenue", monday_revenue),
        ("tuesday_revenue", tuesday_revenue),
    ]:
        if type(value) not in (int, float) or value < 0:
            raise ValueError(f"{field} must be a non-negative number.")

    return {
        "monday_units": monday_units,
        "tuesday_units": tuesday_units,
        "monday_revenue": monday_revenue,
        "tuesday_revenue": tuesday_revenue,
        "units_growth_percent": round(
            (tuesday_units - monday_units) / monday_units * 100, 2
        ),
        "revenue_growth_percent": round(
            (tuesday_revenue - monday_revenue)
            / monday_revenue * 100, 2
        ) if monday_revenue != 0 else None,
        "monday_revenue_per_unit": round(
            monday_revenue / monday_units, 2
        ),
        "tuesday_revenue_per_unit": round(
            tuesday_revenue / tuesday_units, 2
        ),
    }


async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini"
    )

    try:
        analyst = AssistantAgent(
            name="analyst",
            model_client=model_client,
            system_message="""
You are a sales data analyst.

Extract the sales figures from the user's input.
Return ONLY a valid JSON object with these fields:
monday_units, tuesday_units, monday_revenue, tuesday_revenue.

Use numeric values, not strings.
Do not include Markdown fences or extra commentary.
Do not invent missing figures.
""",
        )

        reporter = AssistantAgent(
            name="reporter",
            model_client=model_client,
            system_message="""
You are a business report writer.

Use ONLY the validated JSON data provided to you.
Summarize units growth, revenue growth, and revenue per unit.
Do not change any numeric values.
If a value is null, explain that it cannot be calculated.
""",
        )

        sales_input = """
Monday: 120 units sold, revenue $600.
Tuesday: 150 units sold, revenue $750.
"""
        analyst_result = await analyst.run(task=sales_input)
        analyst_text = analyst_result.messages[-1].content

        print("\n--- ANALYST JSON ---")
        print(analyst_text)

        try:
            sales_data = json.loads(analyst_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Analyst did not return valid JSON."
            ) from exc

        if not isinstance(sales_data, dict):
            raise ValueError("Analyst output must be a JSON object.")

        validated_data = validate_sales_data(sales_data)

        validated_json = json.dumps(
            validated_data,
            indent=2
        )

        print("\n--- VALIDATED DATA ---")
        print(validated_json)
        reporter_result = await reporter.run(
            task=(
                "Create a concise sales report from this validated JSON:\n"
                f"{validated_json}"
            )
        )

        print("\n--- FINAL REPORT ---")
        print(reporter_result.messages[-1].content)

    finally:
        await model_client.close()


asyncio.run(main())