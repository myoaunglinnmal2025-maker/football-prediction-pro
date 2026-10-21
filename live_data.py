"""Optional live adapters for The Odds API v4 and API-Football v3."""
from __future__ import annotations
import os
from typing import Any
import pandas as pd
import requests
import streamlit as st

ODDS_BASE = "https://api.the-odds-api.com/v4"
FOOTBALL_BASE = "https://v3.football.api-sports.io"

def _secret(name: str) -> str:
    value = os.getenv(name, "")
    if value:
        return value.strip()
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""

def get_api_key_status() -> dict[str, str]:
    return {
        "The Odds API": "configured" if _secret("ODDS_API_KEY") else "missing",
        "API-Football": "configured" if _secret("API_FOOTBALL_KEY") else "missing",
    }

def fetch_live_odds(sport_key: str, regions: str = "uk", markets: list[str] | None = None) -> pd.DataFrame:
    key = _secret("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY is missing. Add it to Streamlit Cloud Secrets.")
    response = requests.get(
        f"{ODDS_BASE}/sports/{sport_key}/odds",
        params={"apiKey": key, "regions": regions, "markets": ",".join(markets or ["h2h"]),
                "oddsFormat": "decimal", "dateFormat": "iso"},
        timeout=25,
    )
    if response.status_code == 401:
        raise RuntimeError("The Odds API rejected the key (401). Check your key.")
    if response.status_code == 429:
        raise RuntimeError("The Odds API quota/rate limit was reached.")
    response.raise_for_status()
    rows = []
    for event in response.json():
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "commence_time": event.get("commence_time"),
                        "home_team": event.get("home_team"), "away_team": event.get("away_team"),
                        "bookmaker": book.get("title"), "last_update": market.get("last_update"),
                        "market": market.get("key"), "outcome": outcome.get("name"),
                        "point": outcome.get("point"), "decimal_odds": outcome.get("price")
                    })
    return pd.DataFrame(rows, columns=["commence_time","home_team","away_team","bookmaker","last_update","market","outcome","point","decimal_odds"])

def _football_get(path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    key = _secret("API_FOOTBALL_KEY")
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY is missing. Add it to Streamlit Cloud Secrets.")
    response = requests.get(f"{FOOTBALL_BASE}/{path.lstrip('/')}", params=params,
                            headers={"x-apisports-key": key}, timeout=25)
    if response.status_code == 401:
        raise RuntimeError("API-Football rejected the key. Check the key.")
    if response.status_code == 429:
        raise RuntimeError("API-Football quota/rate limit was reached.")
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(f"API-Football error: {payload['errors']}")
    return payload.get("response", [])

def fetch_live_fixtures() -> pd.DataFrame:
    fixtures = _football_get("fixtures", {"live": "all"})
    rows = []
    for item in fixtures:
        fixture = item.get("fixture", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        rows.append({
            "fixture_id": fixture.get("id"), "kickoff": fixture.get("date"),
            "status": (fixture.get("status") or {}).get("long"),
            "elapsed_min": (fixture.get("status") or {}).get("elapsed"),
            "league": (item.get("league") or {}).get("name"),
            "home_team": (teams.get("home") or {}).get("name"),
            "away_team": (teams.get("away") or {}).get("name"),
            "home_goals": goals.get("home"), "away_goals": goals.get("away"),
            "home_xg": None, "away_xg": None, "statistics_status": "not queried"
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for i, row in df.iterrows():
        try:
            stats = _football_get("fixtures/statistics", {"fixture": int(row["fixture_id"])})
            xg = {}
            for team_stats in stats:
                team_name = (team_stats.get("team") or {}).get("name", "")
                for stat in team_stats.get("statistics") or []:
                    if str(stat.get("type", "")).lower() in {"expected_goals", "expected goals", "xg"} and stat.get("value") is not None:
                        try: xg[team_name] = float(str(stat["value"]).replace("%", ""))
                        except (ValueError, TypeError): pass
            df.at[i, "home_xg"] = xg.get(row["home_team"])
            df.at[i, "away_xg"] = xg.get(row["away_team"])
            df.at[i, "statistics_status"] = "xG available" if xg else "xG not provided for this fixture"
        except Exception as exc:
            df.at[i, "statistics_status"] = f"statistics unavailable: {exc}"
    return df
