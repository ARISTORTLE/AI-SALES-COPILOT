from __future__ import annotations

import json
from typing import Any

import pandas as pd


DEFAULT_MODEL = "grok-4-1-fast-non-reasoning"
XAI_BASE_URL = "https://api.x.ai/v1"


def _table_records(frame: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    sample = frame.head(limit).copy()
    for column in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[column]):
            sample[column] = sample[column].astype(str)
    return sample.to_dict(orient="records")


def build_copilot_payload(
    summary: dict[str, Any],
    insight_cards: list[str],
    weekday_performance: pd.DataFrame,
    product_performance: pd.DataFrame,
    restock_table: pd.DataFrame,
    business_context: str = "",
) -> str:
    payload = {
        "summary": summary,
        "insight_cards": insight_cards,
        "weekday_performance_top": _table_records(weekday_performance, limit=7),
        "product_performance_top": _table_records(product_performance, limit=5),
        "restock_top": _table_records(restock_table, limit=5),
        "business_context": business_context.strip(),
    }
    return json.dumps(payload, indent=2)


def generate_owner_brief(
    api_key: str,
    payload: str,
    model: str = DEFAULT_MODEL,
    focus_prompt: str = "",
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The API client package is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    client = OpenAI(api_key=api_key, base_url=XAI_BASE_URL)

    focus = focus_prompt.strip() or (
        "Focus on practical actions to increase revenue, protect margin, and avoid stockouts."
    )

    instructions = f"""
You are an expert AI sales copilot for Indian small businesses.

You will receive structured analytics from a sales dashboard. Write a concise owner briefing in simple,
practical language. Avoid jargon, avoid sounding academic, and never invent facts that are not in the data.

Your response must have these exact section headings:

## Executive Summary
2 to 3 sentences on what is happening overall.

## What Needs Attention
3 bullet points about the most important risks or sales patterns.

## Recommended Actions
3 bullet points with specific next steps the owner can take this week.

## Suggested Experiment
1 short paragraph describing one practical sales experiment to run next.

Business focus:
{focus}
""".strip()

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=f"Sales analytics payload:\n{payload}",
    )
    return response.output_text.strip()
