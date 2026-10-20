# Football Prediction Pro v3 — deployable research starter

A Streamlit web app for historical match analysis, leakage-aware pre-match features, 1X2 ML probabilities, Poisson goal markets, and chronological holdout evaluation. It is a research tool, not a guaranteed betting system.

## Run on a computer

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app loads a default league/season selection on first visit. Use the sidebar to choose league/seasons and **Reload data and retrain now** to force a fresh request. App-level source data is cached for 24 hours; it refreshes when the app is used after cache expiry.

## Deploy online (Streamlit Community Cloud)

1. Create a GitHub account/repository and upload the contents of this folder (the files must be at repository root, not nested an extra level).
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select the repository and branch, set main file path to `app.py`, and deploy.
4. Open the deployed HTTPS URL on both phone and computer. The same URL works on both; no separate mobile install is needed.
5. Public apps should not contain secrets. If adding a paid data API later, store keys in Streamlit app Secrets and never commit them.

## Scheduled automatic data refresh

Included workflow: `.github/workflows/daily-refresh.yml`. It runs every day at 02:17 UTC and can also be started manually from GitHub → Actions → Daily football data refresh → Run workflow.

To enable it:

1. Push this project to GitHub.
2. In the repository, open **Settings → Actions → General → Workflow permissions** and allow read/write repository contents.
3. Open **Actions** and enable workflows if GitHub prompts you.
4. Run the workflow once manually and check its log for the downloaded row count and latest match date.
5. The workflow commits `data/auto_history.csv` only when the data changes. Repository commits normally trigger a Streamlit Cloud redeploy if the app is connected to that repository.

The daily job refreshes historical results from football-data.co.uk for five European leagues and recent seasons. It does not provide upcoming fixtures, live scores, xG, injuries, lineups or bookmaker odds. The source may publish results with delays or change CSV availability. A scheduled refresh cannot force a source to publish data that it does not yet provide.

## What v3 includes

- Historical results from football-data.co.uk for supported leagues/seasons.
- Team attack/defence and form features built only from matches strictly before each match date.
- Regularized multinomial logistic regression for 1X2 probabilities.
- Chronological holdout metrics and downloadable audit predictions.
- Poisson goal-market estimates for totals, BTTS, Asian handicap settlement and correct scores.
- Fair-odds / expected-return calculator and a daily GitHub Actions refresh workflow.

## Limitations / next production work

- The scheduled dataset is refreshed independently; the app's current model training still loads selected seasons from the source directly.
- Current ML 1X2 and Poisson goal-market outputs are separate models and not jointly calibrated.
- The current backtest is a chronological holdout, not a full rolling-origin evaluation of every market or betting ROI.
- Team availability, xG, fixture feeds and odds require a licensed/reliable provider; none are silently fabricated here.
- For a durable multi-user production service, add a database, model/version registry, automated tests, monitoring, and secrets management.
- Estimates are uncertain and do not guarantee positive returns. Never treat model probabilities as certainty.
