"""
Fetch global news from multiple free RSS/XML feeds using httpx.
No API key required.
"""

import asyncio
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import re

import httpx

# Free RSS feeds — no API key required
RSS_FEEDS = [
    ("Reuters - World",      "https://feeds.reuters.com/reuters/worldNews"),
    ("Reuters - Business",   "https://feeds.reuters.com/reuters/businessNews"),
    ("BBC World",            "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("BBC Business",         "http://feeds.bbci.co.uk/news/business/rss.xml"),
    ("CNBC Top News",        "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("MarketWatch",          "https://feeds.marketwatch.com/marketwatch/topstories"),
    ("Financial Times",      "https://www.ft.com/rss/home/us"),
    ("Al Jazeera",           "https://www.aljazeera.com/xml/rss/all.xml"),
    ("The Guardian World",   "https://www.theguardian.com/world/rss"),
    ("Associated Press",     "https://rsshub.app/apnews/topics/apf-topnews"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; NewsStockApp/1.0; "
        "+https://github.com/emichel99/blackhole)"
    )
}


@dataclass
class NewsItem:
    title: str
    description: str
    url: str
    source: str
    published: Optional[str] = None
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = re.sub(r"\W+", "_", self.title[:60]).strip("_").lower()


def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    clean = re.sub(r"<[^>]+>", " ", text or "")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:500]  # truncate long descriptions


def _parse_rss(content: bytes, source_name: str) -> List[NewsItem]:
    """Parse RSS/Atom XML and return NewsItem list."""
    items: List[NewsItem] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return items

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "media": "http://search.yahoo.com/mrss/",
    }

    # Try RSS 2.0 first
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        desc = _strip_html(item.findtext("description") or "")
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()

        if title and link:
            items.append(NewsItem(
                title=title,
                description=desc,
                url=link,
                source=source_name,
                published=pub,
            ))

    # Try Atom if no RSS items found
    if not items:
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            title = (title_el.text or "").strip() if title_el is not None else ""

            summary_el = entry.find("atom:summary", ns)
            desc = _strip_html((summary_el.text or "") if summary_el is not None else "")

            link_el = entry.find("atom:link[@rel='alternate']", ns) or entry.find("atom:link", ns)
            link = link_el.get("href", "") if link_el is not None else ""

            updated_el = entry.find("atom:updated", ns)
            pub = (updated_el.text or "").strip() if updated_el is not None else ""

            if title and link:
                items.append(NewsItem(
                    title=title,
                    description=desc,
                    url=link,
                    source=source_name,
                    published=pub,
                ))

    return items[:10]  # at most 10 items per feed


async def _fetch_feed(
    client: httpx.AsyncClient,
    name: str,
    url: str,
) -> List[NewsItem]:
    """Fetch a single RSS feed and parse it."""
    try:
        resp = await client.get(url, headers=HEADERS, timeout=8.0, follow_redirects=True)
        resp.raise_for_status()
        return _parse_rss(resp.content, name)
    except Exception:
        return []


async def fetch_all_news(max_items: int = 30) -> List[NewsItem]:
    """
    Fetch news from all configured RSS feeds concurrently.
    Falls back to demo data if no feeds are reachable.
    Returns up to `max_items` deduplicated items.
    """
    async with httpx.AsyncClient() as client:
        tasks = [_fetch_feed(client, name, url) for name, url in RSS_FEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    seen_titles: set = set()
    all_items: List[NewsItem] = []

    for result in results:
        if isinstance(result, list):
            for item in result:
                key = item.title.lower()[:80]
                if key not in seen_titles:
                    seen_titles.add(key)
                    all_items.append(item)
                    if len(all_items) >= max_items:
                        return all_items

    # Fall back to demo data if no live feeds were reachable
    if not all_items:
        return get_demo_news(max_items)

    return all_items


def fetch_all_news_sync(max_items: int = 30) -> List[NewsItem]:
    """Synchronous wrapper around fetch_all_news."""
    return asyncio.run(fetch_all_news(max_items))


# ─────────────────────────────────────────────
# Demo / fallback data
# ─────────────────────────────────────────────

DEMO_NEWS: List[NewsItem] = [
    NewsItem(
        title="Fed holds interest rates steady, signals two cuts in 2025",
        description="The Federal Reserve kept its benchmark interest rate unchanged Wednesday, "
                    "but officials signaled they still expect to cut rates twice before year-end "
                    "despite persistent inflation above the 2% target.",
        url="https://example.com/fed-rates",
        source="Reuters - Business",
        published="Mon, 23 Mar 2026 14:30:00 GMT",
    ),
    NewsItem(
        title="TSMC warns of prolonged chip shortage as Taiwan tensions rise",
        description="Taiwan Semiconductor Manufacturing Co. warned that escalating geopolitical "
                    "tensions in the Taiwan Strait could disrupt global semiconductor supply chains "
                    "for up to 18 months, sending chipmaker stocks lower across Asia and the US.",
        url="https://example.com/tsmc-shortage",
        source="Reuters - World",
        published="Mon, 23 Mar 2026 11:15:00 GMT",
    ),
    NewsItem(
        title="Oil prices surge as OPEC+ announces surprise production cut of 1.5 million bpd",
        description="OPEC+ nations agreed to cut oil output by 1.5 million barrels per day starting "
                    "in April, catching markets off guard. Brent crude jumped 6% to $94 a barrel. "
                    "Airline stocks fell sharply on rising fuel cost concerns.",
        url="https://example.com/opec-cut",
        source="BBC Business",
        published="Mon, 23 Mar 2026 09:00:00 GMT",
    ),
    NewsItem(
        title="Microsoft acquires AI startup for $8.5 billion in largest deal of the year",
        description="Microsoft announced it would acquire AI research firm Cognify for $8.5 billion, "
                    "its largest acquisition since Activision Blizzard. The deal is expected to "
                    "accelerate Microsoft's generative AI capabilities in enterprise software.",
        url="https://example.com/msft-acquisition",
        source="CNBC Top News",
        published="Mon, 23 Mar 2026 08:30:00 GMT",
    ),
    NewsItem(
        title="Bird flu H5N1 spreads to 14 US states, WHO declares health emergency",
        description="The World Health Organization declared a public health emergency of international "
                    "concern after H5N1 avian influenza cases were confirmed in farm workers across "
                    "14 US states. Pharmaceutical and vaccine companies saw shares surge.",
        url="https://example.com/bird-flu",
        source="BBC World",
        published="Sun, 22 Mar 2026 19:45:00 GMT",
    ),
    NewsItem(
        title="Amazon announces 15,000 layoffs as e-commerce growth slows",
        description="Amazon will cut 15,000 jobs globally, primarily in its retail and cloud "
                    "divisions, as the company faces slowing growth and increased competition. "
                    "The announcement sent Amazon shares down 4% in after-hours trading.",
        url="https://example.com/amazon-layoffs",
        source="MarketWatch",
        published="Sun, 22 Mar 2026 16:00:00 GMT",
    ),
    NewsItem(
        title="Russia-Ukraine ceasefire talks collapse, fighting intensifies in eastern regions",
        description="Peace negotiations between Russia and Ukraine broke down in Vienna after "
                    "Ukraine rejected a proposed ceasefire that would have ceded occupied territories. "
                    "European defense stocks rose while energy and grain futures spiked.",
        url="https://example.com/ukraine-ceasefire",
        source="Al Jazeera",
        published="Sun, 22 Mar 2026 13:30:00 GMT",
    ),
    NewsItem(
        title="Apple unveils Vision Pro 2 with 40% lower price, targets mass market",
        description="Apple introduced its second-generation Vision Pro spatial computer at $2,499, "
                    "a 40% reduction from the original model. CEO Tim Cook said the company aims to "
                    "ship 10 million units in the first year. Suppliers in Asia-Pacific rallied.",
        url="https://example.com/apple-vp2",
        source="Reuters - Business",
        published="Sat, 21 Mar 2026 21:00:00 GMT",
    ),
    NewsItem(
        title="California megadrought triggers water rationing in Los Angeles and San Francisco",
        description="California Governor declared a state of emergency as reservoirs hit record lows, "
                    "ordering mandatory 25% water use reductions for residential and commercial users. "
                    "Water utility stocks rose while agriculture-dependent companies fell.",
        url="https://example.com/california-drought",
        source="The Guardian World",
        published="Sat, 21 Mar 2026 15:00:00 GMT",
    ),
    NewsItem(
        title="China's Evergrande collapse triggers $340B debt restructuring, banks brace for losses",
        description="Chinese property giant Evergrande finally collapsed into formal liquidation, "
                    "triggering one of the largest debt restructurings in history. Global banks with "
                    "exposure to Chinese real estate saw shares fall as contagion fears spread.",
        url="https://example.com/evergrande-collapse",
        source="Financial Times",
        published="Sat, 21 Mar 2026 10:00:00 GMT",
    ),
    NewsItem(
        title="Tesla reports record Q1 deliveries but margins fall to five-year low",
        description="Tesla delivered 550,000 vehicles in Q1 2026, a company record, but gross margins "
                    "fell to 12.1% due to aggressive price cuts. Analysts are divided on whether "
                    "volume growth justifies the margin compression.",
        url="https://example.com/tesla-deliveries",
        source="CNBC Top News",
        published="Fri, 20 Mar 2026 22:00:00 GMT",
    ),
    NewsItem(
        title="India surpasses China as world's largest manufacturing hub",
        description="A report from the World Bank confirmed India overtook China as the world's "
                    "largest manufacturing hub by employment in 2025, attracting $180 billion in "
                    "foreign direct investment. Indian stocks hit an all-time high.",
        url="https://example.com/india-manufacturing",
        source="Reuters - World",
        published="Fri, 20 Mar 2026 18:00:00 GMT",
    ),
    NewsItem(
        title="Nvidia unveils Blackwell Ultra GPUs with 10x performance gains for AI training",
        description="Nvidia CEO Jensen Huang unveiled the next-generation Blackwell Ultra GPU platform, "
                    "claiming 10x better performance for large-scale AI model training. Nvidia shares "
                    "hit a new record while AMD fell as competitive gap widened.",
        url="https://example.com/nvidia-blackwell",
        source="MarketWatch",
        published="Fri, 20 Mar 2026 12:00:00 GMT",
    ),
    NewsItem(
        title="Global banking crisis fears resurface as three mid-size European banks fail stress tests",
        description="The European Central Bank revealed that three mid-size lenders failed its annual "
                    "stress tests, raising fears of a broader banking crisis. Financial stocks across "
                    "Europe fell 5-8% while bond yields spiked to 2023 levels.",
        url="https://example.com/eu-banks",
        source="Financial Times",
        published="Thu, 19 Mar 2026 17:00:00 GMT",
    ),
    NewsItem(
        title="SpaceX Starship completes first full orbital mission with crew aboard",
        description="SpaceX's Starship successfully completed its first fully crewed orbital mission, "
                    "landing both the booster and upper stage. The achievement accelerates NASA's "
                    "Artemis moon program timeline and boosted defense and aerospace stocks.",
        url="https://example.com/spacex-starship",
        source="Associated Press",
        published="Thu, 19 Mar 2026 08:00:00 GMT",
    ),
]


def get_demo_news(max_items: int = 30) -> List[NewsItem]:
    """Return demo news articles for environments without internet access."""
    return DEMO_NEWS[:max_items]
