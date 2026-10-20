import streamlit as st
import pandas as pd
from engine import LEAGUES, FEATURES, load_seasons, features, pre_match_dataset, fit_evaluate, markets, ev_binary, ev_asian

st.set_page_config(page_title='Football Prediction Pro | Professional',page_icon='⚽',layout='wide')
st.title('⚽ Football Prediction Pro — Professional Lab')
st.caption('Historical data → leakage-safe features → ML → out-of-sample backtest → goal markets')
st.info('Auto-refresh: source data is fetched when the app runs and cached for 24 hours. For scheduled repository refresh, enable the included GitHub Actions workflow after publishing this project to GitHub.')
@st.cache_data(ttl=86400,show_spinner='Downloading historical match results…')
def get_data(league,seasons):return load_seasons(league,seasons)
@st.cache_data(show_spinner='Building leakage-safe pre-match features…')
def make_features(df):return pre_match_dataset(df)
@st.cache_resource(show_spinner='Training and evaluating model…')
def train(rows):return fit_evaluate(rows)

with st.sidebar:
    st.header('Historical data')
    league=st.selectbox('League',list(LEAGUES))
    years=list(range(2018,2027))
    opts=[f'{y%100:02d}{(y+1)%100:02d}' for y in years]
    seasons=st.multiselect('Seasons',opts,default=opts[-5:-1],help='Select completed seasons; current season may be incomplete.')
    st.caption('Source: football-data.co.uk; internet connection required. Data may have missing/inconsistent rows.')
    start=st.button('Load data and train',type='primary')
if start:
    st.session_state['config']=(league,tuple(seasons))
if 'config' not in st.session_state:
    st.info('Choose a league and seasons in the sidebar, then click **Load data and train**.');st.stop()
lg,ss=st.session_state['config']
try:
    if not ss:raise ValueError('Select at least one season.')
    df=get_data(lg,ss)
    rows=make_features(df)
    model,metrics,audit=train(rows)
except Exception as exc:
    st.error(f'Could not complete training: {exc}');st.stop()

st.caption(f"Data source returned {len(df):,} matches. Latest historical date: {pd.to_datetime(df['Date']).max().date()}. Use the sidebar button to force a reload.")
st.subheader('1 · Historical data and out-of-sample backtest')
c1,c2,c3,c4=st.columns(4)
c1.metric('Historical matches',len(df));c2.metric('Train games',metrics['train_matches']);c3.metric('Test games',metrics['test_matches']);c4.metric('Test accuracy',f"{metrics['accuracy']:.1%}")
c1,c2,c3=st.columns(3)
c1.metric('Test log loss',f"{metrics['log_loss']:.3f}")
c2.metric('Baseline log loss',f"{metrics['baseline_log_loss']:.3f}")
c3.metric('Multiclass Brier',f"{metrics['brier_multiclass']:.3f}")
st.caption(f"Chronological split: training through {metrics['train_until']}; test from {metrics['test_from']}. Test matches were not used to fit ML coefficients. These are test-set metrics, not a live profit guarantee.")
st.download_button('Download backtest predictions CSV',audit.to_csv(index=False).encode('utf-8'),file_name='backtest_predictions.csv',mime='text/csv')
with st.expander('Review backtest rows'):
    st.dataframe(audit.tail(100),hide_index=True)

st.subheader('2 · Match prediction')
teams=sorted(set(df.HomeTeam).union(df.AwayTeam))
c1,c2=st.columns(2)
with c1:home=st.selectbox('Home team',teams,index=0)
with c2:away=st.selectbox('Away team',teams,index=min(1,len(teams)-1))
if home==away:st.warning('Select two different teams.');st.stop()
latest=df.copy() # Latest historical snapshot; no future results used.
f,lh,la=features(latest,home,away)
prob=model.predict_proba(pd.DataFrame([f])[FEATURES])[0]
ml=dict(zip(model.classes_,prob))
st.caption('ML 1X2 probabilities and Poisson goal-market probabilities are separate models; they are not jointly calibrated.')
c1,c2,c3=st.columns(3)
c1.metric('ML Home win',f"{ml['H']:.1%}");c2.metric('ML Draw',f"{ml['D']:.1%}");c3.metric('ML Away win',f"{ml['A']:.1%}")
st.write(f'Estimated expected goals: **{home} {lh:.2f} — {la:.2f} {away}**')
c1,c2=st.columns(2)
with c1:line=st.selectbox('Total goals line',[0.5,1.5,2.5,3.5,4.5],index=2)
with c2:ah=st.selectbox('Home Asian handicap',[x/4 for x in range(-12,13)],index=10)
p,settle,scores=markets(lh,la,line,ah)
st.dataframe(pd.DataFrame([{'Market':k,'Probability':v,'Fair odds (binary)':1/v if v>0 else None} for k,v in p.items()]),hide_index=True)
c1,c2=st.columns(2)
with c1:
    st.markdown('**Asian Handicap settlement probabilities**')
    st.dataframe(pd.DataFrame([{'Settlement':k,'Probability':v} for k,v in sorted(settle.items(),reverse=True)]),hide_index=True)
with c2:
    st.markdown('**Top correct scores**')
    st.dataframe(pd.DataFrame(scores,columns=['Score','Probability']),hide_index=True)

st.subheader('3 · Odds / expected value calculator')
selection=st.selectbox('Selection',list(p)+['Home AH','ML Home win','ML Draw','ML Away win'])
odds=st.number_input('Offered decimal odds',min_value=1.01,max_value=1000.0,value=2.00,step=0.05)
if selection=='Home AH':ev=ev_asian(settle,odds)
elif selection.startswith('ML '):ev=ev_binary(ml[{'ML Home win':'H','ML Draw':'D','ML Away win':'A'}[selection]],odds)
else:ev=ev_binary(p[selection],odds)
st.metric('Expected return per 1 unit stake',f'{ev:+.2%}')
st.warning('Educational research tool, not a verified betting advantage. The goal model is not yet independently calibrated; Asian handicap has pushes/half outcomes. Historical coverage, roster changes and data quality affect estimates.')
