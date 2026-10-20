from __future__ import annotations
import math
import pandas as pd
from sklearn.metrics import log_loss
from model import DixonColes
from markets import score_matrix, market_probs

def rolling_backtest(df, min_train=250, retrain_every=50, half_life_days=365):
    df=df.sort_values("date").reset_index(drop=True)
    rows=[]; model=None
    for i in range(min_train,len(df)):
        train=df.iloc[:i]
        row=df.iloc[i]
        if model is None or (i-min_train)%retrain_every==0:
            model=DixonColes(half_life_days).fit(train)
        if row.home_team not in model.idx or row.away_team not in model.idx:
            continue
        lh,la,rho=model.expected_goals(row.home_team,row.away_team)
        p=market_probs(score_matrix(lh,la,rho))
        y="Home" if row.home_goals>row.away_goals else ("Draw" if row.home_goals==row.away_goals else "Away")
        brier=sum((p[c]-(1 if y==c else 0))**2 for c in ["Home","Draw","Away"])
        rows.append({"date":row.date,"home_team":row.home_team,"away_team":row.away_team,
                     "actual":y,"p_home":p["Home"],"p_draw":p["Draw"],"p_away":p["Away"],
                     "brier":brier})
    out=pd.DataFrame(rows)
    if out.empty: return out, {}
    labels=["Home","Draw","Away"]
    ll=log_loss(out.actual, out[["p_home","p_draw","p_away"]], labels=labels)
    acc=((out[["p_home","p_draw","p_away"]].idxmax(axis=1).map(
        {"p_home":"Home","p_draw":"Draw","p_away":"Away"}))==out.actual).mean()
    return out, {"matches":len(out),"log_loss":float(ll),"brier":float(out.brier.mean()),"accuracy":float(acc)}
