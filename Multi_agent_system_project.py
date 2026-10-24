import asyncio
import json
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

def calculate_sales_metrics(
    monday_units: int,
    tuesday_units: int,
    monday_revenue: float,
    tuesday_revenue: float,
) -> str:
    """Calculate sales growth, revenue growth, and average revenue per unit."""

    if type(monday_units) is not int or type(tuesday_units) is not int:
        raise ValueError("Units sold must be integers.")

    if monday_units <= 0 or tuesday_units <= 0:
        raise ValueError("Units sold must be greater than zero.")

    if monday_revenue < 0 or tuesday_revenue < 0:
        raise ValueError("Revenue cannot be negative.")

    if monday_revenue == 0:
        revenue_growth = None
    else:
        revenue_growth = (
            (tuesday_revenue - monday_revenue)
            / monday_revenue
        ) * 100

    units_growth = (
        (tuesday_units - monday_units)
        / monday_units
    ) * 100

    monday_average = monday_revenue / monday_units
    tuesday_average = tuesday_revenue / tuesday_units

    metrics = {
        "monday_units": monday_units,
        "tuesday_units": tuesday_units,
        "units_growth_percent": round(units_growth, 2),
        "monday_revenue": round(monday_revenue, 2),
        "tuesday_revenue": round(tuesday_revenue, 2),
        "revenue_growth_percent": (
            round(revenue_growth, 2)
            if revenue_growth is not None
            else None
        ),
        "monday_revenue_per_unit": round(monday_average, 2),
        "tuesday_revenue_per_unit": round(tuesday_average, 2),
    }

    return json.dumps(metrics, indent=2)


async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        parallel_tool_calls=False,
    )

    try:
        analyst = AssistantAgent(
            name="analyst",
            model_client=model_client,
            system_message="""
            You are a business data analyst.
            Identify the key facts in the provided sales data.
            Do not invent data or calculate unsupported figures.
            Return a concise factual analysis.
            """,
        )

        calculator = AssistantAgent(
            name="calculator",
            model_client=model_client,
            tools=[calculate_sales_metrics],
            max_tool_iterations=3,
            system_message="""
            You are a business metrics specialist.
            You MUST call calculate_sales_metrics using the
            exact figures supplied in the task.
            Do not calculate metrics mentally.
            Report the returned JSON accurately.
            """,
        )

        reviewer = AssistantAgent(
            name="reviewer",
            model_client=model_client,
            system_message="""
            You are a careful business report reviewer.
            Check the analyst's findings against the original data
            and verify that the metrics match the calculator output.
            Do not invent or change figures.
            If everything is consistent, state APPROVED.
            If something is wrong, state REJECTED and explain why.
            """,
        )

        sales_data = """
        Monday:
        Units sold = 120
        Revenue = $600

        Tuesday:
        Units sold = 150
        Revenue = $750
        """
        analysis_result = await analyst.run(task=sales_data)
        analysis_text = analysis_result.messages[-1].content

        print("\n" + "=" * 50)
        print("ANALYST OUTPUT")
        print("=" * 50)
        print(analysis_text)
        calculation_task = f"""
        Use the calculate_sales_metrics tool on this data.

        {sales_data}

        Return the complete calculated metrics.
        """

        calculation_result = await calculator.run(
            task=calculation_task
        )
        calculation_text = calculation_result.messages[-1].content

        print("\n" + "=" * 50)
        print("CALCULATOR OUTPUT")
        print("=" * 50)
        print(calculation_text)

        review_task = f"""
        Review this business analysis.

        ORIGINAL DATA:
        {sales_data}

        ANALYST FINDINGS:
        {analysis_text}

        CALCULATOR OUTPUT:
        {calculation_text}

        Check the figures and consistency.
        End with APPROVED if everything is consistent,
        otherwise end with REJECTED.
        """

        review_result = await reviewer.run(task=review_task)
        review_text = review_result.messages[-1].content

        print("\n" + "=" * 50)
        print("REVIEWER OUTPUT")
        print("=" * 50)
        print(review_text)
        print("\n" + "=" * 50)
        print("FINAL BUSINESS REPORT")
        print("=" * 50)

        if "APPROVED" in review_text.upper() and "REJECTED" not in review_text.upper():
            print("Status: Reviewer approved the report.")
            print("\nAnalysis:")
            print(analysis_text)
            print("\nCalculated metrics:")
            print(calculation_text)
        else:
            print("Status: Report requires further review.")
            print(review_text)

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
