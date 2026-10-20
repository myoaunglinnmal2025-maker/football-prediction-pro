from __future__ import annotations
import math, joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config import MODEL_FILE

def poisson_logpmf(k, lam):
    lam = max(float(lam), 1e-9)
    return k*math.log(lam)-lam-math.lgamma(k+1)

def dc_tau(x, y, lh, la, rho):
    if x == 0 and y == 0: return 1 - lh*la*rho
    if x == 0 and y == 1: return 1 + lh*rho
    if x == 1 and y == 0: return 1 + la*rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0

class DixonColes:
    def __init__(self, half_life_days=365.0):
        self.half_life_days = half_life_days

    def fit(self, df: pd.DataFrame):
        self.teams = sorted(set(df.home_team) | set(df.away_team))
        self.idx = {t:i for i,t in enumerate(self.teams)}
        n = len(self.teams)
        max_date = pd.to_datetime(df.date).max()
        ages = (max_date - pd.to_datetime(df.date)).dt.days.to_numpy()
        weights = np.exp(-math.log(2)*ages/self.half_life_days)
        hg = df.home_goals.astype(int).to_numpy()
        ag = df.away_goals.astype(int).to_numpy()
        hi = df.home_team.map(self.idx).to_numpy()
        ai = df.away_team.map(self.idx).to_numpy()

        def unpack(p):
            attack = p[:n].copy()
            attack -= attack.mean()
            defense = p[n:2*n]
            home_adv, intercept, rho_raw = p[-3:]
            rho = 0.20*np.tanh(rho_raw)
            return attack, defense, home_adv, intercept, rho

        def loss(p):
            attack, defense, home_adv, intercept, rho = unpack(p)
            lh = np.exp(intercept + home_adv + attack[hi] - defense[ai])
            la = np.exp(intercept + attack[ai] - defense[hi])
            total = 0.0
            for x,y,h,a,w in zip(hg,ag,lh,la,weights):
                tau = dc_tau(int(x),int(y),h,a,rho)
                if tau <= 0: return 1e12
                total -= w*(poisson_logpmf(int(x),h)+poisson_logpmf(int(y),a)+math.log(tau))
            reg = 0.002*(np.square(attack).sum()+np.square(defense).sum())
            return total + reg

        x0 = np.zeros(2*n+3)
        x0[-2] = math.log(max((df.home_goals.sum()+df.away_goals.sum())/(2*len(df)), 0.2))
        res = minimize(loss, x0, method="L-BFGS-B", options={"maxiter":700})
        if not res.success:
            raise RuntimeError("Model fitting failed: " + res.message)
        self.params = res.x
        self.train_end = str(max_date.date())
        self.n_matches = len(df)
        return self

    def expected_goals(self, home, away):
        if home not in self.idx or away not in self.idx:
            raise ValueError("Both teams must exist in the training data.")
        n=len(self.teams); p=self.params
        attack=p[:n].copy(); attack-=attack.mean()
        defense=p[n:2*n]; home_adv,intercept,rho_raw=p[-3:]
        return (
            float(np.exp(intercept+home_adv+attack[self.idx[home]]-defense[self.idx[away]])),
            float(np.exp(intercept+attack[self.idx[away]]-defense[self.idx[home]])),
            float(0.20*np.tanh(rho_raw)),
        )

    def save(self, path=MODEL_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

def load_model(path=MODEL_FILE):
    return joblib.load(path)
