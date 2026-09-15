from pathlib import Path
import pandas as pd, numpy as np, joblib, json, shutil
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,mean_absolute_error,mean_squared_error,r2_score,roc_curve,precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from .model_service import FEATURES,clear,rescore_audience,paths

def _mapping(train,scaler,km):
 c=pd.DataFrame(scaler.inverse_transform(km.cluster_centers_),columns=FEATURES); score=c.sessions_7d+c.listening_minutes_7d+c.discovery_rate+c.days_active_14d-c.skip_rate; names=['Daily Loyalist','Explorer','Binge Listener','Drifting User','Ghost / Low Activity']; order=list(score.sort_values(ascending=False).index); return {cid:names[j] for j,cid in enumerate(order)}
def train_models(uploaded,mode,company=None):
 base=Path(settings.DATA_DIR); md,dd=paths(company); md.mkdir(parents=True,exist_ok=True); dd.mkdir(parents=True,exist_ok=True)
 new=pd.read_csv(uploaded); missing=[f for f in FEATURES if f not in new.columns]
 if missing:raise ValueError('Missing features: '+', '.join(missing))
 if 'churn' not in new.columns:raise ValueError('CSV must contain churn as 0/1.')
 new=new.dropna(subset=FEATURES+['churn']).copy(); new['churn']=new['churn'].astype(int)
 if len(new)<50 or new.churn.nunique()<2:raise ValueError('Need at least 50 rows with both churn classes present.')
 oldpath=dd/'users.csv'; old=pd.read_csv(oldpath) if oldpath.exists() else pd.read_csv(base/'users.csv'); train=new if mode=='replace' else pd.concat([old,new],ignore_index=True); train=train.dropna(subset=FEATURES+['churn']).copy(); X=train[FEATURES]; y=train.churn
 Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42); clf=RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=5,class_weight='balanced',random_state=42,n_jobs=-1); clf.fit(Xtr,ytr); prob=clf.predict_proba(Xte)[:,1]; pred=(prob>=.5).astype(int); fpr,tpr,_=roc_curve(yte,prob); prec,rec,_=precision_recall_curve(yte,prob)
 metrics={'churn_accuracy':float(accuracy_score(yte,pred)),'churn_precision':float(precision_score(yte,pred,zero_division=0)),'churn_recall':float(recall_score(yte,pred,zero_division=0)),'churn_f1':float(f1_score(yte,pred,zero_division=0)),'churn_roc_auc':float(roc_auc_score(yte,prob)),'confusion_matrix':confusion_matrix(yte,pred).tolist(),'roc_curve':{'fpr':fpr.tolist(),'tpr':tpr.tolist()},'pr_curve':{'precision':prec.tolist(),'recall':rec.tolist()}}
 if 'ltv' in train.columns:
  rr=train.dropna(subset=['ltv']); rtr,rte=train_test_split(rr,test_size=.2,random_state=42); reg=RandomForestRegressor(n_estimators=300,max_depth=12,min_samples_leaf=5,random_state=42,n_jobs=-1); reg.fit(rtr[FEATURES],np.log1p(rtr.ltv)); rp=np.expm1(reg.predict(rte[FEATURES])); metrics.update({'ltv_mae':float(mean_absolute_error(rte.ltv,rp)),'ltv_rmse':float(np.sqrt(mean_squared_error(rte.ltv,rp))),'ltv_r2':float(r2_score(rte.ltv,rp))})
 else: reg=joblib.load(Path(settings.MODEL_DIR)/'ltv_model.pkl')['model']; metrics.update({'ltv_mae':None,'ltv_rmse':None,'ltv_r2':None})
 if 'next_action' in train.columns and train.next_action.nunique()>1:
  aa=train.dropna(subset=['next_action']); atr,ate,aytr,ayte=train_test_split(aa[FEATURES],aa.next_action,test_size=.2,stratify=aa.next_action,random_state=42); act=RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=5,class_weight='balanced',random_state=42,n_jobs=-1); act.fit(atr,aytr); metrics['action_accuracy']=float(accuracy_score(ayte,act.predict(ate)))
 else: act=joblib.load(Path(settings.MODEL_DIR)/'action_model.pkl')['model']; metrics['action_accuracy']=None
 scaler=StandardScaler(); Z=scaler.fit_transform(train[FEATURES]); km=KMeans(n_clusters=5,n_init=20,random_state=42).fit(Z); mapping=_mapping(train,scaler,km)
 for f in ['churn_model.pkl','ltv_model.pkl','action_model.pkl','segment_model.pkl']:
  src=md/f
  if src.exists(): b=md/'backups';b.mkdir(exist_ok=True);shutil.copy2(src,b/(f+'.bak'))
 joblib.dump({'model':clf,'features':FEATURES},md/'churn_model.pkl'); joblib.dump({'model':reg,'features':FEATURES},md/'ltv_model.pkl'); joblib.dump({'model':act,'features':FEATURES},md/'action_model.pkl'); joblib.dump({'model':km,'scaler':scaler,'mapping':mapping,'features':FEATURES},md/'segment_model.pkl')
 train.to_csv(dd/'active_training_data.csv',index=False)
 report={'rows':len(train),'features':FEATURES,**metrics,'segments':5,'feature_importance':dict(sorted(zip(FEATURES,clf.feature_importances_),key=lambda x:-x[1])),'source':'uploaded dataset','mode':mode,'generated_at':pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}; (md/'model_report.json').write_text(json.dumps(report,indent=2)); clear(company); rescore_audience(company); clear(company); return report

def test_model(uploaded,company=None):
 m=__import__('predictor.model_service',fromlist=['load']).load(company); df=pd.read_csv(uploaded); miss=[f for f in FEATURES if f not in df.columns]
 if miss:raise ValueError('Missing features: '+', '.join(miss))
 if 'churn' not in df.columns:raise ValueError('Test CSV must contain churn as 0/1.')
 df=df.dropna(subset=FEATURES+['churn']); y=df.churn.astype(int); p=m['churn']['model'].predict_proba(df[FEATURES])[:,1]; pred=(p>=.5).astype(int)
 return {'rows':len(df),'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),'roc_auc':roc_auc_score(y,p),'confusion_matrix':confusion_matrix(y,pred).tolist()}
