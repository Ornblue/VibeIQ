from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from django.conf import settings
FEATURES=['sessions_7d','session_change_7d','listening_minutes_7d','listening_change_7d','skip_rate','save_rate','share_rate','playlist_rate','search_rate','discovery_rate','unique_artists_7d','unique_genres_7d','days_active_14d','avg_session_minutes','subscription_age_days','support_tickets_30d','night_listening_share','completion_rate','avg_daily_sessions','weekend_usage_share','session_gap_hours','artist_concentration','genre_concentration','notification_open_rate','recommendation_click_rate','offline_downloads_30d','playlists_created_30d','social_interactions_30d','device_switch_rate','ad_skip_rate','notification_fatigue','content_diversity']
RATE_FEATURES=['skip_rate','save_rate','share_rate','playlist_rate','search_rate','discovery_rate','night_listening_share','completion_rate']
_cache={}
def company_key(company): return company.slug if company else 'demo'
def paths(company=None):
    if company:
        md=Path(settings.COMPANY_MODEL_DIR)/company.slug; dd=Path(settings.COMPANY_DATA_DIR)/company.slug
        md.mkdir(parents=True,exist_ok=True); dd.mkdir(parents=True,exist_ok=True); return md,dd
    return Path(settings.MODEL_DIR),Path(settings.DATA_DIR)
def load(company=None):
    key=company_key(company)
    if key in _cache:return _cache[key]
    md,dd=paths(company)
    # company models fall back to shipped demo artifacts until the company trains its own model
    src=md if (md/'churn_model.pkl').exists() else Path(settings.MODEL_DIR)
    out={}
    for k in ['churn','ltv','action','segment']:
        out[k]=joblib.load(src/f'{k}_model.pkl')
    scored=dd/'audience_predictions.csv'
    fallback=dd/'users.csv'
    if not scored.exists():
        fallback=Path(settings.DATA_DIR)/'audience_predictions.csv' if (Path(settings.DATA_DIR)/'audience_predictions.csv').exists() else fallback
    out['data']=pd.read_csv(scored if scored.exists() else fallback)
    report=src/'model_report.json'; out['report']=json.loads(report.read_text()) if report.exists() else json.loads((Path(settings.MODEL_DIR)/'model_report.json').read_text())
    _cache[key]=out; return out
def clear(company=None):
    if company:_cache.pop(company_key(company),None)
    else:_cache.clear()
def _risk(p):return 'CRITICAL' if p>=.72 else 'HIGH' if p>=.50 else 'MODERATE' if p>=.25 else 'LOW'
def _prepare(values):
    row={}
    for f in FEATURES:
        v=float(values.get(f,0) or 0)
        if f in RATE_FEATURES and v>1:v/=100
        row[f]=v
    return pd.DataFrame([row])
def _recommend(values,p,ltv,data):
    high=ltv>=data['predicted_ltv'].quantile(.75) if 'predicted_ltv' in data else False
    if p>=.72 and high:return 'Priority retention: personalized re-engagement for a high-value at-risk user.'
    if values.get('skip_rate',0)>.55:return 'Reduce skip-heavy sessions with stronger personalization and better first-result matching.'
    if values.get('discovery_rate',0)<.25:return 'Increase discovery and personalized exploration to rebuild engagement.'
    if values.get('sessions_7d',0)<3:return 'Re-engage with a lightweight personalized entry point and timely reminder.'
    return 'Maintain the experience and use recommendations to grow engagement.'
def predict_one(values,company=None):
    m=load(company); x=_prepare(values)
    p=float(m['churn']['model'].predict_proba(x)[:,1][0]); ltv=float(np.expm1(m['ltv']['model'].predict(x)[0])); action=str(m['action']['model'].predict(x)[0])
    segid=int(m['segment']['model'].predict(m['segment']['scaler'].transform(x))[0]); seg=m['segment']['mapping'].get(segid,'Segment'); health=max(0,min(100,100*(1-p))); risk=_risk(p); data=m['data']
    med=data[FEATURES].median(numeric_only=True); impacts=[]
    for f in FEATURES:
        z=x.copy(); z.loc[0,f]=med[f]; q=float(m['churn']['model'].predict_proba(z)[:,1][0]); impacts.append({'feature':f,'impact':(p-q)*100})
    impacts=sorted(impacts,key=lambda a:abs(a['impact']),reverse=True)[:12]
    sensitivity=[]
    for item in impacts[:8]:
        f=item['feature']; z=x.copy(); z.loc[0,f]=med[f]
        q=float(m['churn']['model'].predict_proba(z)[:,1][0])
        sensitivity.append({'feature':f,'current':round(p*100,3),'counterfactual':round(q*100,3),'delta':round((p-q)*100,3)})
    profile_features=['sessions_7d','listening_minutes_7d','skip_rate','save_rate','discovery_rate','days_active_14d','recommendation_click_rate','content_diversity','completion_rate','notification_open_rate']
    profile=[]
    for f in profile_features:
        series=pd.to_numeric(data[f],errors='coerce').dropna() if f in data else pd.Series([0])
        val=float(values.get(f,0) or 0)
        if len(series)>1:
            strength=float((series<=val).mean())
        else: strength=.5
        profile.append({'feature':f,'strength':round(strength,4),'value':val,'median':float(med[f]) if pd.notna(med[f]) else 0.0})
    return {'user_id':values.get('user_id','NEW_USER'),'churn_probability':p,'health':health,'risk':risk,'ltv':ltv,'next_action':action,'segment':seg,'recommendation':_recommend(values,p,ltv,data),'impacts':impacts,'sensitivity':sensitivity,'profile':profile,'top_positive':[x for x in impacts if x['impact']>0][:5],'top_negative':[x for x in impacts if x['impact']<0][:5]}
def rescore_audience(company=None,raw=None):
    m=load(company); md,dd=paths(company); raw=raw if raw is not None else pd.read_csv(dd/'users.csv') if (dd/'users.csv').exists() else pd.read_csv(Path(settings.DATA_DIR)/'users.csv')
    X=raw[FEATURES]; p=m['churn']['model'].predict_proba(X)[:,1]; ltv=np.expm1(m['ltv']['model'].predict(X)); act=m['action']['model'].predict(X); ids=m['segment']['model'].predict(m['segment']['scaler'].transform(X)); seg=[m['segment']['mapping'].get(int(i),'Segment') for i in ids]
    out=raw.copy(); out['segment_id']=ids; out['segment']=seg; out['churn_probability']=p; out['predicted_ltv']=ltv; out['predicted_next_action']=act; out['engagement_health']=np.clip((1-p)*100,0,100); out['risk']=np.where(p>=.72,'CRITICAL',np.where(p>=.50,'HIGH',np.where(p>=.25,'MODERATE','LOW')))
    out['recommendation']=[_recommend(r,float(pp),float(ll),out) for r,pp,ll in zip(out.to_dict('records'),p,ltv)]
    out.to_csv(dd/'audience_predictions.csv',index=False); _cache[company_key(company)]=m; _cache[company_key(company)]['data']=out; return out
