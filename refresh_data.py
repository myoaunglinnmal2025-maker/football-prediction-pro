"""Refresh recent football-data.co.uk results for supported leagues.
Run locally or via GitHub Actions. Only completed matches with valid scores are retained.
"""
from pathlib import Path
import time
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "auto_history.csv"
LEAGUES = {"E0":"Premier League", "SP1":"La Liga", "D1":"Bundesliga", "I1":"Serie A", "F1":"Ligue 1"}
# Most recent eight season codes; the current season may not yet be published.
SEASONS = [f"{y%100:02d}{(y+1)%100:02d}" for y in range(2018, 2026)] + ["2627"]

def main():
    frames=[]
    errors=[]
    for code, league in LEAGUES.items():
        for season in SEASONS:
            url=f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            try:
                r=requests.get(url, timeout=25, headers={"User-Agent":"FootballPredictionPro/3.0"})
                if r.status_code != 200 or not r.text.strip():
                    continue
                from io import StringIO
                df=pd.read_csv(StringIO(r.text), encoding="latin-1", on_bad_lines="skip")
                needed={"Date","HomeTeam","AwayTeam","FTHG","FTAG"}
                if not needed.issubset(df.columns):
                    continue
                df=df[["Date","HomeTeam","AwayTeam","FTHG","FTAG"]].copy()
                df.columns=["date","home_team","away_team","home_goals","away_goals"]
                df["date"]=pd.to_datetime(df["date"],dayfirst=True,errors="coerce")
                df["home_goals"]=pd.to_numeric(df["home_goals"],errors="coerce")
                df["away_goals"]=pd.to_numeric(df["away_goals"],errors="coerce")
                df=df.dropna(subset=["date","home_team","away_team","home_goals","away_goals"])
                df["home_goals"]=df.home_goals.astype(int); df["away_goals"]=df.away_goals.astype(int)
                df["league"]=league; df["season"]=season
                if len(df): frames.append(df)
                time.sleep(0.15)
            except Exception as e:
                errors.append(f"{league} {season}: {e}")
    if not frames:
        raise RuntimeError("No usable match data downloaded. Check source availability/network.")
    result=pd.concat(frames,ignore_index=True).drop_duplicates(["date","home_team","away_team","league"])
    result=result.sort_values(["date","league","home_team"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(OUT,index=False)
    print(f"Saved {len(result)} matches across {result.league.nunique()} leagues to {OUT}")
    print(f"Latest match date: {result.date.max().date()}")
    if errors: print("Non-fatal source errors:\n"+"\n".join(errors[:10]))

if __name__ == "__main__": main()
