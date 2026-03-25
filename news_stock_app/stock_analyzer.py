"""
Use Claude to analyze a news item and identify:
  1. Impacted stocks / companies (with ticker if known)
  2. Impact direction and magnitude
  3. Event category and severity (for Monte Carlo)
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """\
You are a financial analyst AI specializing in identifying how news events affect stock markets.

When given a news headline and description you will:
1. Identify all publicly-traded companies or ETFs directly or indirectly impacted.
2. For each, specify the ticker symbol (use "UNKNOWN" if unsure), company name, impact direction
   (positive / negative / neutral), and impact magnitude (1-10).
3. Classify the event category as one of:
   geopolitical | natural_disaster | economic | corporate | health | technology | political | social
4. Estimate the event severity on a scale of 1 (minor) to 10 (extreme/crisis-level).
5. Write a brief reasoning (2-3 sentences max).

Respond ONLY with valid JSON matching this schema (no markdown, no extra text):
{
  "stocks": [
    {
      "ticker": "AAPL",
      "company": "Apple Inc.",
      "direction": "negative",
      "magnitude": 7,
      "reason": "Supply chain disruption affects production."
    }
  ],
  "category": "economic",
  "severity": 6,
  "summary": "Brief 1-sentence event summary."
}

If no publicly-traded companies are impacted, return stocks as an empty array.
"""


@dataclass
class StockImpact:
    ticker: str
    company: str
    direction: str      # positive | negative | neutral
    magnitude: int      # 1–10
    reason: str


@dataclass
class AnalysisResult:
    stocks: List[StockImpact]
    category: str
    severity: float
    summary: str
    raw_json: dict


async def analyze_news_async(title: str, description: str) -> AnalysisResult:
    """Async version using streaming for reliability."""
    prompt = f"Headline: {title}\n\nDescription: {description}"

    async_client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    async with async_client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = await stream.get_final_message()

    text = next(
        (block.text for block in message.content if block.type == "text"),
        "{}"
    )

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON substring
        start = text.find("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end]) if start != -1 else {}

    stocks = [
        StockImpact(
            ticker=s.get("ticker", "UNKNOWN"),
            company=s.get("company", "Unknown Company"),
            direction=s.get("direction", "neutral"),
            magnitude=int(s.get("magnitude", 5)),
            reason=s.get("reason", ""),
        )
        for s in data.get("stocks", [])
    ]

    return AnalysisResult(
        stocks=stocks,
        category=data.get("category", "default"),
        severity=float(data.get("severity", 5)),
        summary=data.get("summary", title),
        raw_json=data,
    )


def analysis_to_dict(result: AnalysisResult) -> dict:
    return {
        "stocks": [
            {
                "ticker": s.ticker,
                "company": s.company,
                "direction": s.direction,
                "magnitude": s.magnitude,
                "reason": s.reason,
            }
            for s in result.stocks
        ],
        "category": result.category,
        "severity": result.severity,
        "summary": result.summary,
    }
