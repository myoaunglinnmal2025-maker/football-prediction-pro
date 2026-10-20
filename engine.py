import math
import numpy as np
import pandas as pd

LEAGUES = {'Premier League':'E0','La Liga':'SP1','Bundesliga':'D1','Serie A':'I1','Ligue 1':'F1'}
FEATURES = ['home_attack','home_defense','away_attack','away_defense','home_form','away_form','home_games','away_games']

def load_seasons(league, seasons):
    parts=[]
    for season in seasons:
        url=f'https://www.football-data.co.uk/mmz4281/{season}/{LEAGUES[league]}.csv'
        try:
            df=pd.read_csv(url, on_bad_lines='skip', encoding='latin-1')
            required={'Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR'}
            if not required.issubset(df.columns):
                raise ValueError('Required match columns not found')
            df=df[list(required)].copy()
            df['Date']=pd.to_datetime(df['Date'],dayfirst=True,errors='coerce')
            for col in ['FTHG','FTAG']:
                df[col]=pd.to_numeric(df[col],errors='coerce')
            df=df.dropna(subset=list(required))
            df=df[df.FTR.isin(['H','D','A'])]
            df['season']=season
            parts.append(df)
        except Exception as exc:
            raise RuntimeError(f'Cannot download {league} {season}: {exc}') from exc
    if not parts: raise ValueError('No seasons selected')
    return pd.concat(parts,ignore_index=True).sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)

def strength(history, team, venue, prior=5.0, max_matches=20):
    if venue=='home':
        sub=history[history.HomeTeam==team].tail(max_matches)
        scored=sub.FTHG.to_numpy(dtype=float); conceded=sub.FTAG.to_numpy(dtype=float)
    else:
        sub=history[history.AwayTeam==team].tail(max_matches)
        scored=sub.FTAG.to_numpy(dtype=float); conceded=sub.FTHG.to_numpy(dtype=float)
    n=len(sub)
    if n:
        # Exponentially favor more recent games, but shrink sparse samples.
        weights=np.exp(np.linspace(-1.0,0.0,n))
        gs=float(np.average(scored,weights=weights))
        gc=float(np.average(conceded,weights=weights))
        points=np.where(scored>conceded,3,np.where(scored==conceded,1,0))
        form=float(np.average(points,weights=weights)/3)
    else: gs=gc=prior; form=0.5
    shrink=n/(n+6)
    return shrink*gs+(1-shrink)*prior, shrink*gc+(1-shrink)*prior, shrink*form+(1-shrink)*0.5, n

def features(history, home, away):
    if len(history):
        base_home=float(history.FTHG.mean()); base_away=float(history.FTAG.mean())
    else: base_home,base_away=1.45,1.15
    base_home=max(0.2,base_home); base_away=max(0.2,base_away)
    ha,hd,hf,hn=strength(history,home,'home',base_home)
    aa,ad,af,an=strength(history,away,'away',base_away)
    row=dict(zip(FEATURES,[ha,hd,aa,ad,hf,af,hn,an]))
    # attack strength multiplied by opponent's conceded relative to league average
    lh=np.clip(ha*ad/base_away,0.2,4.5)
    la=np.clip(aa*hd/base_home,0.2,4.5)
    return row,float(lh),float(la)

def pre_match_dataset(df, min_history=30):
    # Strict date-grouped construction: same-day results cannot leak into features.
    frames=[]
    for date, group in df.groupby('Date',sort=True):
        history=df[df.Date<date]
        if len(history)<min_history: continue
        for _,m in group.iterrows():
            f,lh,la=features(history,m.HomeTeam,m.AwayTeam)
            f.update(Date=date,HomeTeam=m.HomeTeam,AwayTeam=m.AwayTeam,FTR=m.FTR,FTHG=int(m.FTHG),FTAG=int(m.FTAG),lambda_home=lh,lambda_away=la)
            frames.append(f)
    return pd.DataFrame(frames)

def score_matrix(lh,la,n=12):
    def pois(lam,k):return math.exp(-lam+k*math.log(lam)-math.lgamma(k+1))
    p=np.array([pois(lh,i) for i in range(n+1)])[:,None]*np.array([pois(la,j) for j in range(n+1)])[None,:]
    return p/p.sum()

def asian_result(diff,line):
    if abs(line*4-round(line*4))>1e-7:raise ValueError('Handicap must be quarter-goal line')
    if abs(line*2-round(line*2))>1e-7:
        low=math.floor(line*2)/2; high=math.ceil(line*2)/2
        return (asian_result(diff,low)+asian_result(diff,high))/2
    v=diff+line
    return 1 if v>1e-8 else (-1 if v< -1e-8 else 0)

def markets(lh,la,total_line=2.5,ah_line=-0.5):
    p=score_matrix(lh,la)
    i,j=np.indices(p.shape)
    out={'Home win':float(p[i>j].sum()),'Draw':float(p[i==j].sum()),'Away win':float(p[i<j].sum()),'Over':float(p[i+j>total_line].sum()),'Under':float(p[i+j<total_line].sum()),'BTTS Yes':float(p[(i>0)&(j>0)].sum()),'BTTS No':float(p[(i==0)|(j==0)].sum())}
    settlements={k:0.0 for k in [-1,-0.5,0,0.5,1]}
    for x in range(p.shape[0]):
        for y in range(p.shape[1]):settlements[asian_result(x-y,ah_line)]+=float(p[x,y])
    top=sorted([(f'{x}-{y}',float(p[x,y])) for x in range(p.shape[0]) for y in range(p.shape[1])],key=lambda t:t[1],reverse=True)[:10]
    return out,settlements,top

def ev_binary(p,odds):return p*(odds-1)-(1-p)
def ev_asian(settlements,odds):return sum(prob*((odds-1)*max(k,0)+min(k,0)) for k,prob in settlements.items())

def fit_evaluate(data,split=0.8):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, accuracy_score
    if len(data)<120:raise ValueError('Need at least 120 pre-match rows for training and testing.')
    # Time ordered, no shuffle; split by date so matches on the boundary do not cross.
    dates=sorted(data.Date.unique()); cut=dates[max(1,int(len(dates)*split))-1]
    train=data[data.Date<=cut]; test=data[data.Date>cut]
    if len(test)<20 or train.FTR.nunique()!=3:raise ValueError('Insufficient class diversity or test games.')
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,C=0.3))
    model.fit(train[FEATURES],train.FTR)
    pred=model.predict_proba(test[FEATURES]); classes=list(model.classes_)
    target=np.array([[int(y==c) for c in classes] for y in test.FTR])
    brier=float(np.mean(np.sum((pred-target)**2,axis=1)))
    baseline=np.tile(train.FTR.value_counts(normalize=True).reindex(classes,fill_value=0).to_numpy(),(len(test),1))
    result={'train_matches':len(train),'test_matches':len(test),'train_until':str(pd.Timestamp(cut).date()),'test_from':str(pd.Timestamp(test.Date.min()).date()),'log_loss':float(log_loss(test.FTR,pred,labels=classes)),'brier_multiclass':brier,'accuracy':float(accuracy_score(test.FTR,model.predict(test[FEATURES]))),'baseline_log_loss':float(log_loss(test.FTR,baseline,labels=classes))}
    audit=test[['Date','HomeTeam','AwayTeam','FTR']].copy()
    for k in ['H','D','A']:audit[f'P_{k}']=pred[:,classes.index(k)]
    audit['Predicted']=model.predict(test[FEATURES])
    return model,result,audit
