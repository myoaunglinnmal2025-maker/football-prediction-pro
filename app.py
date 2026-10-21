import streamlit as st
import pandas as pd
from engine import LEAGUES, FEATURES, load_seasons, features, pre_match_dataset, fit_evaluate, markets, ev_binary, ev_asian
from betting import no_vig_1x2, expected_roi, fractional_kelly, closing_line_value, betting_backtest
from walkforward import evaluate, calibration_table

st.set_page_config(page_title='Football Prediction Pro | Professional',page_icon='⚽',layout='wide')
st.title('⚽ Football Prediction Pro v4 — Betting Intelligence Lab')
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
    league=st.selectbox('League',list(LEAGUES)+['Champions League (CSV upload)'])
    ucl_file=st.file_uploader('Champions League results CSV (optional)',type=['csv']) if league.startswith('Champions') else None
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
    if lg.startswith('Champions'):
        if ucl_file is None: raise ValueError('Upload a Champions League CSV first. Required columns: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR.')
        df=pd.read_csv(ucl_file)
        required={'Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR'}
        if not required.issubset(df.columns):raise ValueError('UCL CSV missing required columns')
        df=df[list(required)].copy()
        df['Date']=pd.to_datetime(df.Date,dayfirst=True,errors='coerce')
        for c in ['FTHG','FTAG']:df[c]=pd.to_numeric(df[c],errors='coerce')
        df=df.dropna(subset=list(required)).sort_values('Date').reset_index(drop=True)
    else:df=get_data(lg,ss)
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


st.divider()
st.subheader('4 · Bookmaker margin, risk & closing-line analysis')
st.caption('Enter your own quoted odds. The app does NOT fetch live bookmaker odds or place bets.')
o1,o2,o3=st.columns(3)
with o1:odd_h=st.number_input('Home odds',min_value=1.01,value=2.30,step=.05,key='oh')
with o2:odd_d=st.number_input('Draw odds',min_value=1.01,value=3.30,step=.05,key='od')
with o3:odd_a=st.number_input('Away odds',min_value=1.01,value=3.10,step=.05,key='oa')
market_odds={'H':odd_h,'D':odd_d,'A':odd_a}
novig=no_vig_1x2(market_odds)
margin=sum(1/x for x in market_odds.values())-1
risk_fraction=st.slider('Kelly fraction (risk simulation)',0.0,0.5,0.25,0.05)
max_stake=st.slider('Maximum bankroll allocation',0.0,0.05,0.01,0.005)
compare=pd.DataFrame([{'Selection':k,'Model probability':ml[k], 'No-vig market probability':novig[k],
                        'Offered odds':market_odds[k],'Model EV':expected_roi(ml[k],market_odds[k]),
                        'Fractional Kelly allocation':fractional_kelly(ml[k],market_odds[k],risk_fraction,max_stake)}
                      for k in ['H','D','A']])
st.metric('Bookmaker overround',f'{margin:.2%}')
st.dataframe(compare,hide_index=True)
st.caption('Model and market probabilities may disagree because of model error, missing team news and stale data. Kelly is illustrative, not a stake recommendation.')
closing=st.number_input('Closing odds for selected 1X2 market (manual)',min_value=1.01,value=2.20,step=.05)
st.metric('Illustrative closing line value (Home)',f'{closing_line_value(odd_h,closing):+.2%}')

st.subheader('5 · Expanding-window walk-forward validation')
st.caption('Re-trains on earlier match dates and tests on later date blocks. Slower than the simple holdout above.')
if st.button('Run walk-forward validation'):
    with st.spinner('Running expanding-window evaluation…'):
        try:
            wf,wfm=evaluate(rows)
            st.session_state['wf_audit']=wf
            st.session_state['wf_metrics']=wfm
        except Exception as exc:st.error(f'Walk-forward evaluation failed: {exc}')
if 'wf_audit' in st.session_state:
    wf=st.session_state['wf_audit'];wfm=st.session_state['wf_metrics']
    a,b,c,d=st.columns(4)
    a.metric('Test matches',wfm['n_test']);b.metric('Training folds',wfm['folds'])
    c.metric('Log loss',f"{wfm['log_loss']:.3f}");d.metric('Brier',f"{wfm['brier']:.3f}")
    st.dataframe(calibration_table(wf),hide_index=True)
    st.download_button('Download walk-forward audit',wf.to_csv(index=False),file_name='walkforward.csv',mime='text/csv')
    st.subheader('6 · Hypothetical fixed-odds paper backtest')
    st.warning('This is a scenario, NOT historical betting ROI: odds are assumed constant and are not historical pre-match quotes. It cannot establish a real betting edge.')
    s=st.selectbox('Paper selection',['H','D','A'])
    fixed=st.number_input('Hypothetical decimal odds',min_value=1.01,value=2.0,step=.05)
    edge=st.slider('Minimum expected ROI threshold',0.0,0.25,0.03,0.01)
    if st.button('Simulate paper bets'):
        simulated,report=betting_backtest(wf,s,fixed,edge)
        st.json(report)
        if not simulated.empty:st.download_button('Download paper simulation',simulated.to_csv(index=False),file_name='paper_simulation.csv',mime='text/csv')


st.divider()
st.subheader('7 · Live odds & live match statistics (API integrations)')
st.caption('Optional provider connections. Odds/xG coverage depends on your API plan, league and fixture.')
with st.expander('Configure live API connections', expanded=True):
    st.markdown("""
    In Streamlit Cloud → **Manage app → Settings → Secrets**, add:
    ```toml
    ODDS_API_KEY = "your_the_odds_api_key"
    API_FOOTBALL_KEY = "your_api_football_key"
    ```
    Replace placeholders with real keys. Never commit API keys to GitHub.
    """)
    from live_data import fetch_live_odds, fetch_live_fixtures, get_api_key_status
    st.write("API key status:", get_api_key_status())
    sport_key = st.selectbox("Odds sport key", [
        "soccer_epl", "soccer_uefa_champs_league", "soccer_spain_la_liga",
        "soccer_italy_serie_a", "soccer_germany_bundesliga", "soccer_france_ligue_one"
    ])
    regions = st.selectbox("Odds regions", ["uk", "eu", "us", "au"])
    requested_markets = st.multiselect("Odds markets", ["h2h", "totals", "btts", "spreads"], default=["h2h", "totals"])
    ca, cb = st.columns(2)
    with ca:
        if st.button("Fetch live odds"):
            try:
                with st.spinner("Fetching current odds…"):
                    st.session_state["live_odds_df"] = fetch_live_odds(sport_key, regions, requested_markets)
            except Exception as exc:
                st.error(f"Live odds request failed: {exc}")
    with cb:
        if st.button("Fetch live fixtures / available xG"):
            try:
                with st.spinner("Fetching live fixtures and available statistics…"):
                    st.session_state["live_fixtures_df"] = fetch_live_fixtures()
            except Exception as exc:
                st.error(f"Live fixtures request failed: {exc}")
    if "live_odds_df" in st.session_state:
        live_odds_df = st.session_state["live_odds_df"]
        st.markdown("**Current odds**")
        if live_odds_df.empty:
            st.info("No odds returned. Check sport key, region, market coverage and provider quota.")
        else:
            st.dataframe(live_odds_df, use_container_width=True, hide_index=True)
            st.download_button("Download odds CSV", live_odds_df.to_csv(index=False), "live_odds.csv", "text/csv")
    if "live_fixtures_df" in st.session_state:
        live_fixtures_df = st.session_state["live_fixtures_df"]
        st.markdown("**Live fixtures and xG, where provider returns it**")
        if live_fixtures_df.empty:
            st.info("No live fixtures returned at the moment.")
        else:
            st.dataframe(live_fixtures_df, use_container_width=True, hide_index=True)
            st.download_button("Download live fixtures CSV", live_fixtures_df.to_csv(index=False), "live_fixtures.csv", "text/csv")
