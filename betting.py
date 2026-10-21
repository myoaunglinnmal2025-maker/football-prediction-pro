"""Pure analytics functions. No live bookmaker connection or wager placement."""
from __future__ import annotations
import math
import pandas as pd


def implied_probability(odds: float) -> float:
    if not math.isfinite(odds) or odds <= 1: raise ValueError('Decimal odds must exceed 1')
    return 1.0 / odds


def no_vig_1x2(odds: dict[str,float]) -> dict[str,float]:
    if set(odds) != {'H','D','A'}: raise ValueError('Provide H, D and A odds')
    raw={k:implied_probability(v) for k,v in odds.items()}
    total=sum(raw.values())
    return {k:v/total for k,v in raw.items()}


def expected_roi(probability: float, odds: float) -> float:
    if not 0 <= probability <= 1: raise ValueError('Probability must be in [0,1]')
    implied_probability(odds)
    return probability*odds-1


def fractional_kelly(probability: float, odds: float, fraction:float=0.25, cap:float=0.02) -> float:
    if not 0 <= fraction <= 1 or not 0 <= cap <= 1: raise ValueError('Invalid risk settings')
    edge=expected_roi(probability,odds)
    return min(cap,max(0.0, fraction*edge/(odds-1)))


def closing_line_value(open_odds:float, closing_odds:float) -> float:
    implied_probability(open_odds); implied_probability(closing_odds)
    return open_odds/closing_odds-1


def max_drawdown(returns:list[float]) -> float:
    equity=1.0; peak=1.0; worst=0.0
    for r in returns:
        if r < -1: raise ValueError('Return below -100%')
        equity*=1+r
        peak=max(peak,equity)
        worst=max(worst,1-equity/peak)
    return worst


def binary_settlement(selection:str,home_goals:int,away_goals:int,line:float=2.5)->bool:
    if selection=='H': return home_goals>away_goals
    if selection=='D': return home_goals==away_goals
    if selection=='A': return home_goals<away_goals
    if selection=='BTTS Yes': return home_goals>0 and away_goals>0
    if selection=='BTTS No': return home_goals==0 or away_goals==0
    if selection=='Over': return home_goals+away_goals>line
    if selection=='Under': return home_goals+away_goals<line
    raise ValueError('Unknown selection')


def betting_backtest(predictions:pd.DataFrame, selection:str, odds:float, threshold:float=0.03, line:float=2.5):
    """Paper-only fixed-odds scenario, NOT historical ROI: offered odds are hypothetical."""
    implied_probability(odds)
    key={'H':'p_home','D':'p_draw','A':'p_away','BTTS Yes':'p_btts_yes',
         'BTTS No':'p_btts_no','Over':'p_over','Under':'p_under'}[selection]
    needed={key,'FTHG','FTAG'}
    if not needed.issubset(predictions.columns): raise ValueError('Missing prediction or result columns')
    audit=[]
    for _,r in predictions.iterrows():
        edge=expected_roi(float(r[key]),odds)
        if edge<=threshold:continue
        hg,ag=int(r.FTHG),int(r.FTAG)
        if selection in ('Over','Under') and float(line).is_integer() and hg+ag==line:
            pnl=0.0
        else:
            pnl=odds-1 if binary_settlement(selection,hg,ag,line) else -1.0
        audit.append({'date':r.get('Date',''),'selection':selection,'model_p':r[key],
                      'assumed_odds':odds,'expected_roi':edge,'unit_pnl':pnl})
    out=pd.DataFrame(audit)
    if out.empty:return out,{'bets':0,'roi':0.0,'profit_units':0.0,'max_drawdown':0.0}
    return out,{'bets':len(out),'roi':float(out.unit_pnl.mean()),'profit_units':float(out.unit_pnl.sum()),
                'max_drawdown':max_drawdown(out.unit_pnl.mul(0.01).tolist())}
