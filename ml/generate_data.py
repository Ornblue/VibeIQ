from pathlib import Path
import numpy as np, pandas as pd, joblib, json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

RNG=np.random.default_rng(20260915)
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; MODELS=ROOT/'model_artifacts'
DATA.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
N=15000
regions=['Mumbai','Delhi','Bengaluru','Hyderabad','Pune','Kolkata','Chennai','Ahmedabad','Jaipur','Bhubaneswar','Kochi','Lucknow']
languages=['Hindi','English','Punjabi','Marathi','Tamil','Telugu','Bengali','Kannada','Malayalam','Gujarati']
subs=['Free','Premium','Family']

def clip(x,a,b): return float(np.clip(x,a,b))
rows=[]
for i in range(N):
    arche=int(RNG.choice(6,p=[.22,.18,.18,.16,.14,.12]))
    base_sessions=[5,11,4,8,2,7][arche]
    sessions=max(0,int(RNG.poisson(base_sessions)))
    minutes=max(0,float(RNG.gamma(2.4,17 if arche!=4 else 8)))
    skip=clip(RNG.beta(2+(arche in [4,5])*2,7-(arche==1)*2),.01,.95)
    save=clip(RNG.beta(2.4+(arche==1)*2,12),.005,.8)
    share=clip(RNG.beta(1.8+(arche==1),20),.001,.55)
    playlist=clip(RNG.beta(2+(arche in [1,3]),16),.001,.65)
    search=clip(RNG.beta(2+(arche==1)*2,9),.01,.85)
    discovery=clip(RNG.beta(3+(arche==1)*3,7),.01,.95)
    artists=max(1,int(RNG.poisson([3,11,2,7,1,6][arche])+1))
    genres=max(1,int(RNG.poisson([2,6,2,4,1,5][arche])+1))
    days=min(14,max(1,int(RNG.normal([10,13,7,11,3,9][arche],2))))
    avg_session=clip(minutes/max(sessions,1),1,180)
    age=max(1,int(RNG.lognormal(4.7,1.1)))
    tickets=int(RNG.poisson(.30 if arche!=4 else .9))
    night=clip(RNG.beta(2,8)+(.10 if arche==2 else 0),.01,.9)
    completion=clip(1-skip-RNG.normal(.05,.03),.05,.99)
    session_change=float(RNG.normal([-.05,.08,-.22,.03,-.45,-.12][arche],.16))
    listening_change=float(RNG.normal([-.04,.10,-.25,.02,-.50,-.08][arche],.20))

    # Additional audience intelligence features.
    avg_daily_sessions=clip(sessions/max(days,1),0,8)
    weekend_share=clip(RNG.beta(4 if arche in [1,2] else 3,5),.02,.98)
    session_gap_hours=clip(RNG.lognormal(2.5 if arche!=4 else 3.6,.65),.2,96)
    artist_concentration=clip(RNG.beta(3 if arche in [0,4] else 2.0,5 if arche==1 else 3.5),.01,.99)
    genre_concentration=clip(RNG.beta(3 if arche in [0,4] else 2.0,5),.01,.99)
    notification_open_rate=clip(RNG.beta(2.5+(arche==3),7),.01,.95)
    recommendation_click_rate=clip(RNG.beta(3+(arche==1)*2,6),.01,.95)
    offline_downloads= max(0,int(RNG.poisson([1,5,1,3,0,2][arche])))
    playlists_created=max(0,int(RNG.poisson([1,4,0,3,0,2][arche])))
    social_interactions=max(0,int(RNG.poisson([1,5,1,3,0,2][arche])))
    device_switch_rate=clip(RNG.beta(2+(arche==1),8),.01,.8)
    ad_skip_rate=clip(RNG.beta(2+(arche==4)*2,6),.01,.95)
    notification_fatigue=clip(RNG.beta(2+(arche==4)*2,6),.01,.95)
    content_diversity=clip((0.55*discovery+0.25*(artists/15)+0.20*(genres/8))+RNG.normal(0,.04),.01,.99)

    # Synthetic outcome mechanism: disengagement, friction and weak discovery raise churn.
    logit=(-2.35 -.20*sessions + .95*skip -1.8*save -1.0*share -.32*days
           +1.65*max(-session_change,0)+1.35*max(-listening_change,0)+.22*tickets
           +.95*(days<=4)+.65*(completion<.45)+.55*notification_fatigue
           -.75*recommendation_click_rate-.55*discovery+.012*session_gap_hours
           +.45*ad_skip_rate-.55*content_diversity + RNG.normal(0,.42))
    p=1/(1+np.exp(-logit)); churn=int(RNG.random()<p)
    ltv=max(0,125+6.7*minutes+95*save+135*share+38*days+22*artists+28*playlists_created
           +55*recommendation_click_rate*10-165*p+RNG.normal(0,75))
    action=RNG.choice(['PLAY','SEARCH','SKIP','SAVE','SHARE','CREATE_PLAYLIST','LEAVE_APP'],p=[.39,.14,.10,.11,.06,.07,.13])
    if p>.72: action='LEAVE_APP'
    elif recommendation_click_rate>.65 and discovery>.60: action='SEARCH'
    elif save>.22: action='SAVE'
    elif skip>.55 or ad_skip_rate>.65: action='SKIP'
    elif playlist>.16 or playlists_created>=4: action='CREATE_PLAYLIST'
    elif share>.12: action='SHARE'
    rows.append([f'U{i+1:06d}',RNG.choice(regions),RNG.choice(languages),RNG.choice(subs,p=[.56,.31,.13]),
                 sessions,session_change,minutes,listening_change,skip,save,share,playlist,search,discovery,artists,genres,days,avg_session,age,tickets,night,completion,
                 avg_daily_sessions,weekend_share,session_gap_hours,artist_concentration,genre_concentration,notification_open_rate,recommendation_click_rate,offline_downloads,playlists_created,social_interactions,device_switch_rate,ad_skip_rate,notification_fatigue,content_diversity,
                 churn,ltv,action])

cols=['user_id','region','language','subscription','sessions_7d','session_change_7d','listening_minutes_7d','listening_change_7d','skip_rate','save_rate','share_rate','playlist_rate','search_rate','discovery_rate','unique_artists_7d','unique_genres_7d','days_active_14d','avg_session_minutes','subscription_age_days','support_tickets_30d','night_listening_share','completion_rate','avg_daily_sessions','weekend_usage_share','session_gap_hours','artist_concentration','genre_concentration','notification_open_rate','recommendation_click_rate','offline_downloads_30d','playlists_created_30d','social_interactions_30d','device_switch_rate','ad_skip_rate','notification_fatigue','content_diversity','churn','ltv','next_action']
df=pd.DataFrame(rows,columns=cols)
df.to_csv(DATA/'users.csv',index=False)

# 120k event-level observations for richer analytics / portfolio storytelling.
ev=[]
for _,r in df.sample(60000,replace=True,random_state=42).iterrows():
    typ=RNG.choice(['play','search','skip','save','share','playlist','notification_open','recommendation_click','download'],p=[.45,.11,.12,.08,.05,.05,.05,.06,.03])
    ev.append([r.user_id,pd.Timestamp('2026-08-01')+pd.Timedelta(hours=int(RNG.integers(0,24*45))),typ,r.language,RNG.integers(1,360),r.region,r.subscription])
pd.DataFrame(ev,columns=['user_id','timestamp','event_type','language','duration_seconds','region','subscription']).to_csv(DATA/'streaming_events.csv',index=False)

FEATURES=['sessions_7d','session_change_7d','listening_minutes_7d','listening_change_7d','skip_rate','save_rate','share_rate','playlist_rate','search_rate','discovery_rate','unique_artists_7d','unique_genres_7d','days_active_14d','avg_session_minutes','subscription_age_days','support_tickets_30d','night_listening_share','completion_rate','avg_daily_sessions','weekend_usage_share','session_gap_hours','artist_concentration','genre_concentration','notification_open_rate','recommendation_click_rate','offline_downloads_30d','playlists_created_30d','social_interactions_30d','device_switch_rate','ad_skip_rate','notification_fatigue','content_diversity']
X=df[FEATURES]; y=df.churn
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
clf=RandomForestClassifier(n_estimators=80,max_depth=10,min_samples_leaf=5,class_weight='balanced',random_state=42,n_jobs=-1); clf.fit(Xtr,ytr)
proba=clf.predict_proba(Xte)[:,1]; pred=(proba>=.5).astype(int)
reg=RandomForestRegressor(n_estimators=80,max_depth=12,min_samples_leaf=5,random_state=42,n_jobs=-1); reg.fit(Xtr,np.log1p(df.loc[Xtr.index,'ltv']))
lp=np.expm1(reg.predict(Xte)); true=df.loc[Xte.index,'ltv']
axtr,axte,aytr,ayte=train_test_split(X,df.next_action,test_size=.2,stratify=df.next_action,random_state=42)
actclf=RandomForestClassifier(n_estimators=80,max_depth=10,min_samples_leaf=5,class_weight='balanced',random_state=42,n_jobs=-1); actclf.fit(axtr,aytr); ap=actclf.predict(axte)
scaler=StandardScaler(); Z=scaler.fit_transform(X); km=KMeans(n_clusters=6,n_init=20,random_state=42); labels=km.fit_predict(Z)
cent=pd.DataFrame(scaler.inverse_transform(km.cluster_centers_),columns=FEATURES,index=range(6)); score=cent.sessions_7d.rank()+cent.listening_minutes_7d.rank()+cent.discovery_rate.rank()+cent.days_active_14d.rank()-cent.skip_rate.rank(); order=list(score.sort_values(ascending=False).index); names=['Daily Loyalist','Explorer','Binge Listener','Community Builder','Drifting User','Ghost / Low Activity']; mapping={c:names[j] for j,c in enumerate(order)}
df['segment_id']=labels; df['segment']=df.segment_id.map(mapping); df['churn_probability']=clf.predict_proba(X)[:,1]; df['predicted_ltv']=np.expm1(reg.predict(X)); df['predicted_next_action']=actclf.predict(X); df['engagement_health']=np.clip(100*(1-df.churn_probability),0,100); df['risk']=pd.cut(df.churn_probability,bins=[-1,.25,.5,.72,2],labels=['LOW','MODERATE','HIGH','CRITICAL']).astype(str)
def rec(r):
    if r.churn_probability>=.72 and r.predicted_ltv>=df.predicted_ltv.quantile(.75): return 'Priority retention: personalized re-engagement for a high-value at-risk user'
    if r.skip_rate>.55 or r.ad_skip_rate>.65: return 'Improve first-result matching and reduce skip-heavy sessions'
    if r.discovery_rate<.25 or r.content_diversity<.30: return 'Increase discovery and broaden personalized content exposure'
    if r.sessions_7d<3 or r.session_change_7d<-.25: return 'Re-engage with a lightweight personalized entry point'
    if r.notification_fatigue>.65: return 'Reduce notification pressure and optimize message timing'
    if r.save_rate<.08: return 'Surface save-worthy collections and personalized playlists'
    return 'Maintain experience and grow engagement'
df['recommendation']=df.apply(rec,axis=1); df.to_csv(DATA/'audience_predictions.csv',index=False)
train=df[FEATURES+['churn','ltv','next_action']].sample(1200,random_state=42); train.to_csv(DATA/'training_template.csv',index=False)
test=df[FEATURES+['churn','ltv','next_action']].sample(600,random_state=7); test.to_csv(DATA/'test_template.csv',index=False)
for name,obj in [('churn_model.pkl',{'model':clf,'features':FEATURES}),('ltv_model.pkl',{'model':reg,'features':FEATURES}),('action_model.pkl',{'model':actclf,'features':FEATURES}),('segment_model.pkl',{'model':km,'scaler':scaler,'mapping':mapping,'features':FEATURES})]: joblib.dump(obj,MODELS/name)
fpr,tpr,_=roc_curve(yte,proba); pr,rc,_=precision_recall_curve(yte,proba)
roc_points={'fpr':fpr[::max(1,len(fpr)//80)].tolist(),'tpr':tpr[::max(1,len(tpr)//80)].tolist()}; pr_points={'precision':pr[::max(1,len(pr)//80)].tolist(),'recall':rc[::max(1,len(rc)//80)].tolist()}
ltv_sample=pd.DataFrame({'actual':true.values,'predicted':lp}).sample(min(400,len(true)),random_state=42).to_dict('records')
report={'rows':N,'events':60000,'features':FEATURES,'feature_count':len(FEATURES),'churn_rate':float(y.mean()),'churn_accuracy':float(accuracy_score(yte,pred)),'churn_precision':float(precision_score(yte,pred)),'churn_recall':float(recall_score(yte,pred)),'churn_f1':float(f1_score(yte,pred)),'churn_roc_auc':float(roc_auc_score(yte,proba)),'confusion_matrix':confusion_matrix(yte,pred).tolist(),'ltv_mae':float(mean_absolute_error(true,lp)),'ltv_rmse':float(mean_squared_error(true,lp)**.5),'ltv_r2':float(r2_score(true,lp)),'action_accuracy':float(accuracy_score(ayte,ap)),'segments':6,'feature_importance':dict(sorted(zip(FEATURES,clf.feature_importances_),key=lambda x:-x[1])),'segment_counts':df.segment.value_counts().to_dict(),'action_counts':df.next_action.value_counts().to_dict(),'risk_counts':df.risk.value_counts().to_dict(),'roc_curve':roc_points,'pr_curve':pr_points,'ltv_sample':ltv_sample,'generated_at':'2026-09-15','note':'Synthetic streaming-platform audience data for demonstration; not private platform data.'}
(MODELS/'model_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'rows':N,'events':60000,'features':len(FEATURES),'accuracy':report['churn_accuracy'],'roc_auc':report['churn_roc_auc'],'ltv_r2':report['ltv_r2'],'action_accuracy':report['action_accuracy']},indent=2))
