"""
News → Stock Impact → Monte Carlo
FastAPI backend
"""

import asyncio
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from news_fetcher import fetch_all_news, NewsItem
from stock_analyzer import analyze_news_async, analysis_to_dict
from monte_carlo import run_simulation, result_to_dict

app = FastAPI(title="News Stock Impact Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory cache (news + analysis results)
_cache: dict = {
    "news": [],
    "analyzed": {},   # item_id → {analysis, monte_carlo}
    "last_fetch": None,
}


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

@app.get("/", response_class=HTMLResponse)
async def index():
    with open(TEMPLATES_DIR / "index.html") as f:
        return f.read()


@app.get("/api/news")
async def get_news(refresh: bool = False):
    """Return cached news feed (refresh forces a re-fetch)."""
    if refresh or not _cache["news"]:
        items = await fetch_all_news(max_items=30)
        _cache["news"] = [
            {
                "id": item.id,
                "title": item.title,
                "description": item.description,
                "url": item.url,
                "source": item.source,
                "published": item.published,
            }
            for item in items
        ]
    return {"items": _cache["news"], "count": len(_cache["news"])}


@app.get("/api/analyze/{item_id}")
async def analyze_item(item_id: str):
    """
    Analyze a single news item:
      - Identify impacted stocks via Claude
      - Run Monte Carlo simulation
    Returns cached result if available.
    """
    # Check cache
    if item_id in _cache["analyzed"]:
        return _cache["analyzed"][item_id]

    # Find the news item
    item = next((n for n in _cache["news"] if n["id"] == item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="News item not found. Fetch news first.")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not set. Please configure your API key.",
        )

    # Run Claude analysis
    analysis = await analyze_news_async(item["title"], item["description"])
    analysis_dict = analysis_to_dict(analysis)

    # Run Monte Carlo simulation
    mc = run_simulation(
        category=analysis.category,
        severity=analysis.severity,
        days_elapsed=0,
        simulations=10_000,
        horizon_days=30,
    )
    mc_dict = result_to_dict(mc)

    result = {
        "item_id": item_id,
        "news": item,
        "analysis": analysis_dict,
        "monte_carlo": mc_dict,
    }

    _cache["analyzed"][item_id] = result
    return result


@app.get("/api/analyze-batch")
async def analyze_batch(limit: int = 5):
    """
    Analyze the first `limit` news items concurrently.
    Returns a list of analysis results.
    """
    if not _cache["news"]:
        await get_news()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not set.",
        )

    items_to_analyze = [
        n for n in _cache["news"][:limit]
        if n["id"] not in _cache["analyzed"]
    ]

    async def _analyze_one(item: dict):
        try:
            analysis = await analyze_news_async(item["title"], item["description"])
            analysis_dict = analysis_to_dict(analysis)
            mc = run_simulation(
                category=analysis.category,
                severity=analysis.severity,
            )
            mc_dict = result_to_dict(mc)
            result = {
                "item_id": item["id"],
                "news": item,
                "analysis": analysis_dict,
                "monte_carlo": mc_dict,
            }
            _cache["analyzed"][item["id"]] = result
            return result
        except Exception as e:
            return {"item_id": item["id"], "error": str(e)}

    tasks = [_analyze_one(item) for item in items_to_analyze]
    results = await asyncio.gather(*tasks)

    # Include already-cached items
    all_results = list(results) + [
        _cache["analyzed"][n["id"]]
        for n in _cache["news"][:limit]
        if n["id"] in _cache["analyzed"]
        and n["id"] not in [r.get("item_id") for r in results]
    ]

    return {"results": all_results, "count": len(all_results)}


@app.get("/api/monte-carlo")
async def monte_carlo_custom(
    category: str = "default",
    severity: float = 5.0,
    days_elapsed: int = 0,
    simulations: int = 10_000,
):
    """Run a custom Monte Carlo simulation."""
    severity = max(1.0, min(10.0, severity))
    days_elapsed = max(0, days_elapsed)
    simulations = min(50_000, max(100, simulations))

    result = run_simulation(
        category=category,
        severity=severity,
        days_elapsed=days_elapsed,
        simulations=simulations,
    )
    return result_to_dict(result)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "api_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "cached_news": len(_cache["news"]),
        "cached_analyses": len(_cache["analyzed"]),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
