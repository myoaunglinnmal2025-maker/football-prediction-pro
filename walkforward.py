"""Expanding-window, date-grouped model evaluation; never fit on future dates."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from engine import FEATURES

def evaluate(rows,initial_fraction=.60,step_dates=35):
    dates=sorted(pd.to_datetime(rows.Date).unique())
    if len(dates)<100: raise ValueError('Need at least 100 unique match dates')
    begin=max(1,int(len(dates)*initial_fraction))
    records=[]
    for start in range(begin,len(dates),step_dates):
        cut=dates[start-1]
        end=dates[min(start+step_dates,len(dates))-1]
        train=rows[rows.Date<=cut]
        test=rows[(rows.Date>cut)&(rows.Date<=end)]
        if test.empty or train.FTR.nunique()!=3:continue
        m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=.3))
        m.fit(train[FEATURES],train.FTR)
        p=m.predict_proba(test[FEATURES]); classes=list(m.classes_)
        for (_,r),pr in zip(test.iterrows(),p):
            d={'Date':r.Date,'HomeTeam':r.HomeTeam,'AwayTeam':r.AwayTeam,'FTR':r.FTR,
               'FTHG':int(r.FTHG),'FTAG':int(r.FTAG),'train_cutoff':str(pd.Timestamp(cut).date()),
               'lambda_home':r.lambda_home,'lambda_away':r.lambda_away}
            for c in ['H','D','A']:d['p_'+c.lower()]=float(pr[classes.index(c)])
            records.append(d)
    out=pd.DataFrame(records)
    if out.empty:raise ValueError('No evaluation folds')
    p=out[['p_a','p_d','p_h']].to_numpy()
    result={'n_test':len(out),'folds':int(out.train_cutoff.nunique()),
            'log_loss':float(log_loss(out.FTR,p,labels=['A','D','H'])),
            'brier':float(np.mean(sum((out['p_'+c.lower()].to_numpy()-(out.FTR==c).astype(float).to_numpy())**2 for c in ['H','D','A']))),
            'accuracy':float((out[['p_h','p_d','p_a']].idxmax(axis=1).str[-1].str.upper()==out.FTR).mean())}
    return out,result

def calibration_table(audit,bins=8):
    frames=[]
    for cls in ['H','D','A']:
        p=audit['p_'+cls.lower()]
        edges=np.linspace(0,1,bins+1)
        groups=pd.cut(p,edges,include_lowest=True)
        temp=pd.DataFrame({'prob':p,'outcome':(audit.FTR==cls).astype(int),'bucket':groups})
        t=temp.groupby('bucket',observed=True).agg(n=('outcome','size'),predicted=('prob','mean'),observed=('outcome','mean')).reset_index()
        t['outcome']=cls;frames.append(t)
    return pd.concat(frames,ignore_index=True)
