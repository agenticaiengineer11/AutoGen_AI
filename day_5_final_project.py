import asyncio
import json
import math
import os
import re
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

STATE_FILE = "day5_agent_states.json"

def create_model_client():
    return OpenAIChatCompletionClient(
        model="gpt-4o-mini"
    )

def create_agents(model_client):

    analyst = AssistantAgent(
        name="analyst",
        model_client=model_client,
        system_message="""
You are a business data analyst.

Extract these fields from the sales data:
- monday_units
- tuesday_units
- monday_revenue
- tuesday_revenue

Return ONLY a valid JSON object.
Use numbers, not numeric strings.
Do not invent missing data.
Do not use Markdown code fences.
""",
    )

    reviewer = AssistantAgent(
        name="reviewer",
        model_client=model_client,
        system_message="""
You are a financial report reviewer.

Review the supplied sales data and calculated metrics.

Check:
1. Units growth percentage.
2. Revenue growth percentage.
3. Revenue per unit for both days.
4. Whether the supplied metrics match the original figures.

The Python validation process has already checked data types
and calculated the metrics.

Do not claim a correct calculation is incorrect.
Do not invent errors.
Explain any actual inconsistency you identify.

End your response with exactly one of these lines:
REVIEW_VERDICT: APPROVED
REVIEW_VERDICT: REJECTED
""",
    )

    reporter = AssistantAgent(
        name="reporter",
        model_client=model_client,
        system_message="""
You are a professional business report writer.

Create a concise sales report using the validated data
and review information provided in the task.

Include:
- Units sold on Monday and Tuesday.
- Revenue on Monday and Tuesday.
- Units growth percentage.
- Revenue growth percentage.
- Revenue per unit for both days.
- Final report status.

Use the provided numeric values without changing them.
If a value is null, explain that it could not be calculated.
Do not invent facts.
""",
    )

    return analyst, reviewer, reporter

async def save_agent_states(agents):

    states = {}

    for agent in agents:
        states[agent.name] = await agent.save_state()

    with open(STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(states, file, indent=2)

    print(f"\nAgent states saved to {STATE_FILE}")

async def load_agent_states(agents):

    if not os.path.exists(STATE_FILE):
        print("\nNo previous state found. Starting fresh.")
        return

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            states = json.load(file)

        for agent in agents:
            if agent.name in states:
                await agent.load_state(states[agent.name])

        print("Previous agent states restored successfully.")

    except (OSError, json.JSONDecodeError, ValueError, KeyError) as error:
        print(f"Could not restore previous state: {error}")
        print("Continuing with the current agent instances.")

def parse_json_output(text):
    cleaned_text = text.strip()

    cleaned_text = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    )

    cleaned_text = re.sub(
        r"\s*```$",
        "",
        cleaned_text,
    )

    try:
        data = json.loads(cleaned_text)

    except json.JSONDecodeError as error:
        raise ValueError(
            "Analyst did not return valid JSON."
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            "Analyst output must be a JSON object."
        )

    return data
def validate_sales_data(data):

    required_fields = [
        "monday_units",
        "tuesday_units",
        "monday_revenue",
        "tuesday_revenue",
    ]
    for field in required_fields:

        if field not in data:
            raise ValueError(
                f"Missing required field: {field}"
            )

    monday_units = data["monday_units"]
    tuesday_units = data["tuesday_units"]

    monday_revenue = data["monday_revenue"]
    tuesday_revenue = data["tuesday_revenue"]
    if type(monday_units) is not int or monday_units <= 0:
        raise ValueError(
            "monday_units must be a positive integer."
        )

    if type(tuesday_units) is not int or tuesday_units <= 0:
        raise ValueError(
            "tuesday_units must be a positive integer."
        )
    for field, value in [
        ("monday_revenue", monday_revenue),
        ("tuesday_revenue", tuesday_revenue),
    ]:

        if type(value) not in (int, float):
            raise ValueError(
                f"{field} must be a number."
            )

        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"{field} must be finite and non-negative."
            )
    units_growth = (
        (tuesday_units - monday_units)
        / monday_units
        * 100
    )

    revenue_growth = (
        (tuesday_revenue - monday_revenue)
        / monday_revenue
        * 100
        if monday_revenue != 0
        else None
    )

    monday_revenue_per_unit = (
        monday_revenue / monday_units
    )

    tuesday_revenue_per_unit = (
        tuesday_revenue / tuesday_units
    )
    return {
        "monday_units": monday_units,
        "tuesday_units": tuesday_units,
        "monday_revenue": monday_revenue,
        "tuesday_revenue": tuesday_revenue,
        "units_growth_percent": round(units_growth, 2),
        "revenue_growth_percent": (
            round(revenue_growth, 2)
            if revenue_growth is not None
            else None
        ),
        "monday_revenue_per_unit": round(
            monday_revenue_per_unit, 2
        ),
        "tuesday_revenue_per_unit": round(
            tuesday_revenue_per_unit, 2
        ),
    }

async def main():

    model_client = create_model_client()

    try:
        analyst, reviewer, reporter = create_agents(
            model_client
        )

        agents = [analyst, reviewer, reporter]

        await load_agent_states(agents)

        sales_input = """
        Monday: 120 units sold, revenue $600.
        Tuesday: 150 units sold, revenue $750.
        """

        print("\n" + "=" * 45)
        print("ANALYST")
        print("=" * 45)

        analyst_result = await analyst.run(
            task=(
                "Extract the sales figures from this data:\n"
                f"{sales_input}"
            )
        )

        analyst_text = analyst_result.messages[-1].content

        print(analyst_text)
        print("\n" + "=" * 45)
        print("VALIDATION")
        print("=" * 45)

        sales_data = parse_json_output(
            analyst_text
        )

        validated_data = validate_sales_data(
            sales_data
        )

        validated_json = json.dumps(
            validated_data,
            indent=2,
        )

        print(validated_json)

        print("\n" + "=" * 45)
        print("REVIEWER")
        print("=" * 45)

        review_task = f"""
Original sales data:
{sales_input}

Validated metrics calculated by Python:
{validated_json}

Independently review the metrics.
Identify actual mathematical inconsistencies, if any.
Finish with REVIEW_VERDICT: APPROVED or
REVIEW_VERDICT: REJECTED.
"""

        reviewer_result = await reviewer.run(
            task=review_task
        )

        review_text = reviewer_result.messages[-1].content

        print(review_text)
        verdict_match = re.search(
            r"REVIEW_VERDICT:\s*(APPROVED|REJECTED)\b",
            review_text,
            re.IGNORECASE,
        )

        if verdict_match:
            reviewer_verdict = (
                verdict_match.group(1).upper()
            )
        else:
            reviewer_verdict = "REJECTED"

        print("\n" + "=" * 45)
        print("FINAL BUSINESS REPORT")
        print("=" * 45)

        reporter_task = f"""
Create a final business report using the data below.

Validated metrics:
{validated_json}

Reviewer feedback:
{review_text}

Reviewer verdict:
{reviewer_verdict}

Use the reviewer verdict as the report status.
Do not change any validated numeric values.
"""

        reporter_result = await reporter.run(
            task=reporter_task
        )

        report_text = reporter_result.messages[-1].content

        print(f"\nStatus: {reviewer_verdict}")
        print(report_text)
        report_data = {
            "sales_input": sales_input,
            "validated_metrics": validated_data,
            "reviewer_verdict": reviewer_verdict,
            "review": review_text,
            "final_report": report_text,
        }

        with open(
            "final_business_report.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report_data,
                file,
                indent=2,
            )

        print(
            "\nFinal report saved to final_business_report.json"
        )
        await save_agent_states(agents)

    finally:
        await model_client.close()
if __name__ == "__main__":
    asyncio.run(main())