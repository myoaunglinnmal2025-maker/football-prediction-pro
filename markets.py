import math
import numpy as np
from model import dc_tau

def poisson(lam,k):
    return math.exp(-lam + k*math.log(lam) - math.lgamma(k+1))

def score_matrix(lh, la, rho=0.0, max_goals=12):
    m=np.zeros((max_goals+1,max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            m[i,j]=poisson(lh,i)*poisson(la,j)*dc_tau(i,j,lh,la,rho)
    m=np.maximum(m,0); return m/m.sum()

def asian_outcome(diff,line):
    x=diff+line
    if abs(x-round(x))<1e-9: return 1 if x>0 else (-1 if x<0 else 0)
    if abs(x*2-round(x*2))<1e-9: return 1 if x>0 else -1
    lo=math.floor(line*2)/2; hi=math.ceil(line*2)/2
    return (asian_outcome(diff,lo)+asian_outcome(diff,hi))/2

def market_probs(m,total_line=2.5,handicap=0.0):
    n=m.shape[0]
    ph=sum(m[i,j] for i in range(n) for j in range(n) if i>j)
    pd=sum(m[i,i] for i in range(n))
    pa=1-ph-pd
    po=sum(m[i,j] for i in range(n) for j in range(n) if i+j>total_line)
    pb=sum(m[i,j] for i in range(1,n) for j in range(1,n))
    settle={-1:0.0,-0.5:0.0,0:0.0,0.5:0.0,1:0.0}
    for i in range(n):
        for j in range(n):
            settle[asian_outcome(i-j,handicap)]+=m[i,j]
    return {"Home":ph,"Draw":pd,"Away":pa,"Over":po,"Under":1-po,
            "BTTS Yes":pb,"BTTS No":1-pb,"AH":settle}

def top_scores(m,k=10):
    rows=[]
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            rows.append((f"{i}-{j}",float(m[i,j])))
    return sorted(rows,key=lambda x:x[1],reverse=True)[:k]
