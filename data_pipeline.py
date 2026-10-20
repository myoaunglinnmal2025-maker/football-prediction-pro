from __future__ import annotations
from pathlib import Path
from io import StringIO
import pandas as pd
import requests
from config import MATCHES_CSV, FIXTURES_CSV

REQUIRED = ["date","home_team","away_team","home_goals","away_goals"]
ALIASES = {
    "Date":"date","HomeTeam":"home_team","AwayTeam":"away_team",
    "FTHG":"home_goals","FTAG":"away_goals",
    "home_xg":"home_xg","away_xg":"away_xg",
    "B365H":"odds_home","B365D":"odds_draw","B365A":"odds_away",
}

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k:v for k,v in ALIASES.items() if k in df.columns}).copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    for c in ["home_goals","away_goals","home_xg","away_xg","odds_home","odds_draw","odds_away"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED).sort_values("date").reset_index(drop=True)
    return df

def load_csv(source) -> pd.DataFrame:
    return normalize(pd.read_csv(source))

def download_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return load_csv(StringIO(r.text))

def save_matches(df: pd.DataFrame):
    MATCHES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MATCHES_CSV, index=False)

def load_saved_matches() -> pd.DataFrame:
    if not MATCHES_CSV.exists():
        raise FileNotFoundError("No historical dataset saved yet.")
    df = pd.read_csv(MATCHES_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df

def save_fixtures(df: pd.DataFrame):
    FIXTURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIXTURES_CSV, index=False)
