# Football Prediction Pro v4 — Betting Intelligence Lab

A Streamlit research application for football match probabilities and **hypothetical** betting analytics. Built on the v3 architecture. **Not a verified profitable strategy, not live odds, no automated betting.**

## Supported competitions

- Premier League (E0), La Liga (SP1), Bundesliga (D1), Serie A (I1), Ligue 1 (F1): historical match results downloaded from `football-data.co.uk` (provider availability varies).
- Champions League: user-uploaded CSV only. No UCL automatic data feed.

The app does not use licensed live xG, injury, lineups or bookmaker odds. Current 1X2 uses pre-match rolling features + logistic regression; goal markets use a separate Poisson score model. The two are **not jointly calibrated**. An unused experimental Dixon–Coles module from v3 remains in the repository; it is not the active app's prediction engine.

## Deploy to existing Streamlit app

1. **Back up your v3 GitHub repository** or create a new branch.
2. Download and unzip v4. **Upload the contents of the `football_prediction_pro_v4` directory, not the directory itself**, to the **root** of the same GitHub repository. Replace matching files and commit. Preserve `.github/workflows/daily-refresh.yml` (hidden `.github` directory).
3. In Streamlit Community Cloud, your existing app should redeploy when the repository's main branch changes. Confirm **Main file path** is `app.py` and branch is correct. If not, Manage app → Settings / Deploy settings and adjust.
4. If an error occurs, open **Manage app → Logs**. Check `requirements.txt`, `app.py` path and GitHub commit. You can revert the commit if needed.
5. In the sidebar, choose a league and seasons and click **Load data and train**. For Champions League, upload a CSV then click the same button. Next click **Run walk-forward validation** to see the calibration table and paper simulation.

## Local install and tests

Requires Python 3.11+:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
streamlit run app.py
```

## Data and automatic refresh

`refresh_data.py` and `.github/workflows/daily-refresh.yml` are carried forward from v3. The workflow attempts to fetch league match results once per day and commit `data/auto_history.csv` to the repository. **This does not automatically fetch live bookmaker odds, UCL matches, or retrain persisted model files.** The web app reads football-data.co.uk when you load/train and caches source data for 24 hours. Github Actions schedules can be delayed or disabled by GitHub; confirm the Actions tab shows successful runs. Repository write permission is needed for committing data. On a public repository, never commit API keys or private data.

## UCL CSV format

CSV columns: `Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR`.

- `Date` should be unambiguous, preferably `DD/MM/YYYY` for this import; `FTR` must be H, D, A.
- Only completed matches. Aim for **multiple seasons and at least 120 post-feature matches**. UEFA team-name variations should be normalized before upload.
- UCL group/knockout stage effects, extra-time, penalties and neutral venues are not modeled. **Use 90-minute regulation-time results only**.
- The provided `sample_matches.csv` is merely a format sample, insufficient for model training.

## Evaluation / limitations

- Date-grouped pre-match features prevent same-day match-result leakage.
- Simple holdout uses chronological train/test dates; v4 optional expanding-window validation repeatedly refits only on earlier match dates.
- Walk-forward validation is more realistic but still subject to source bias, omitted variables, market selection and repeated tuning. Calibration table is diagnostic, not calibration correction.
- Fixed-odds paper backtest uses **the same hypothetical odds for all matches**; its ROI is **not realized historical ROI**. For a real ROI study, archive actual bookmaker prices at the pre-match decision time, including vig and timestamps, and account for limits/fees.
- Kelly fraction is capped and purely illustrative. Positive model EV does not establish a reliable advantage. Never treat predicted probabilities as guaranteed outcomes.
- No API keys, betting execution or wagering integrations are provided.

## File guide

- `app.py`: Streamlit dashboard
- `engine.py`: historical data, leakage-aware rolling features, 1X2 model, goal markets
- `betting.py`: odds normalization, EV, capped Kelly, CLV and paper simulation
- `walkforward.py`: expanding-window evaluation and calibration diagnostics
- `tests/test_betting.py`: automated unit tests
- `refresh_data.py`: scheduled league-results downloader (from v3)
- `.github/workflows/daily-refresh.yml`: GitHub Actions schedule (from v3)

## Next steps for production-grade research

1. Licensed, timestamped bookmaker odds + odds history and line movements.
2. Licensed xG, injuries, team news, UCL fixture feed, team ID normalization.
3. Store timestamped predictions, model version, data snapshot and actual results in durable database.
4. Fully walk-forward calibration and settlement tests for each market, including quarter Asian handicaps.
5. Monitor calibration drift, data freshness, outages, bookmaker rule differences and closing-line value.


## Live odds and live xG API integrations

Optional adapters use The Odds API v4 for current bookmaker odds and API-Football v3 for live fixtures and statistics where available. API coverage, pricing, quotas and xG support vary by plan and league; no provider guarantees xG for every fixture.

In Streamlit Cloud **Manage app → Settings → Secrets**, add TOML with your own keys:
```toml
ODDS_API_KEY = "your_the_odds_api_key"
API_FOOTBALL_KEY = "your_api_football_key"
```
Never commit real keys to GitHub. The app fetches data only when you press the relevant button. Requests consume provider quota. This is read-only and does not place bets.

Docs: https://the-odds-api.com/liveapi/guides/v4/ and https://www.api-football.com/documentation-v3
