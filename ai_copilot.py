from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


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


def _groq_chat_completion(
    api_key: str,
    system_instructions: str,
    user_content: str,
    model: str = DEFAULT_MODEL,
) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.4,
    }

    request = Request(
        url=f"{GROQ_BASE_URL}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AI-Sales-CoPilot/1.0 (+Streamlit; Python urllib)",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Groq API request failed: {detail or exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Groq API: {exc.reason}") from exc

    choices = parsed.get("choices", [])
    if not choices:
        raise RuntimeError("Groq API returned no choices.")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError("Groq API returned an empty response.")
    return content.strip()


def generate_owner_brief(
    api_key: str,
    payload: str,
    model: str = DEFAULT_MODEL,
    focus_prompt: str = "",
) -> str:
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

    return _groq_chat_completion(
        api_key=api_key,
        system_instructions=instructions,
        user_content=f"Sales analytics payload:\n{payload}",
        model=model,
    )


def generate_action_roadmap(
    api_key: str,
    payload: str,
    model: str = DEFAULT_MODEL,
    focus_prompt: str = "",
) -> dict[str, Any]:
    focus = focus_prompt.strip() or (
        "Focus on practical weekly actions that improve revenue, profit quality, and operational discipline."
    )

    instructions = f"""
You are an expert AI sales operating advisor.

You will receive structured analytics from a sales dashboard. Return only valid JSON with this exact shape:

{{
  "headline": "short string",
  "priority": "short string",
  "steps": [
    {{
      "phase": "Fix Now | Grow Next | Scale Later",
      "title": "short string",
      "why": "1 to 2 sentences",
      "actions": ["action 1", "action 2", "action 3"],
      "impact": "short string"
    }}
  ]
}}

Rules:
- Return exactly 3 steps.
- Be specific and practical.
- Do not include markdown fences.
- Do not invent facts not supported by the payload.
- Keep wording founder-friendly and visual.

Business focus:
{focus}
""".strip()

    response = _groq_chat_completion(
        api_key=api_key,
        system_instructions=instructions,
        user_content=f"Sales analytics payload:\n{payload}",
        model=model,
    )

    try:
        return json.loads(response)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Groq roadmap returned invalid JSON.") from exc
