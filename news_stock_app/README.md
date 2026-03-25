# News Stock Impact Analyzer

A web app that fetches global news, identifies which stocks are impacted, and estimates the probability of each event continuing using Monte Carlo simulation.

## Features

- **Global news feed** — pulls from 10+ free RSS sources (Reuters, BBC, CNBC, MarketWatch, etc.)
- **AI stock analysis** — Claude Opus 4.6 identifies impacted companies, impact direction (▲/▼), and magnitude
- **Monte Carlo simulation** — 10,000-path simulation estimates P(event still active) at 7, 14, and 30 days
- **Interactive dashboard** — click any article to see analysis; auto-analyze top 5

## Setup

```bash
cd news_stock_app
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Run

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/news?refresh=false` | Fetch news feed |
| `GET /api/analyze/{item_id}` | Analyze a single article |
| `GET /api/analyze-batch?limit=5` | Analyze top N articles concurrently |
| `GET /api/monte-carlo?category=economic&severity=7` | Custom Monte Carlo run |
| `GET /api/health` | Health check |

## Monte Carlo Model

Each event is modeled with:
- **Category-specific half-life** — how long similar events typically last (e.g. geopolitical: ~30 days, corporate: ~7 days)
- **Severity multiplier** — higher severity extends expected duration
- **Daily stochastic shocks** — random variation in resolution rate
- **10,000 simulated paths** — smooth probability estimates

The output is P(event still ongoing) at each future day up to 30 days.
