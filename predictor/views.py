from pathlib import Path
import json, pandas as pd, numpy as np
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import FileResponse, JsonResponse
from django.shortcuts import render, redirect
from django.utils.text import slugify
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from .forms import PredictionForm,PredictionFileForm,TrainingForm,FeedbackForm,LabeledExampleForm,TestForm,CompanyRegisterForm,LoginForm,ChatForm,FEATURES
from .models import Company,Membership,TrainingRun,PredictionFeedback,UserPrediction,Dataset
from .model_service import load,predict_one,rescore_audience
from .training_service import train_models,test_model

def _company(request): return getattr(getattr(request.user,'membership',None),'company',None)
def _df(company): return load(company)['data']
def _json(v): return json.dumps(v,default=str)

def home_redirect(request): return redirect('dashboard') if request.user.is_authenticated else redirect('login')

def register(request):
    if request.user.is_authenticated:return redirect('dashboard')
    form=CompanyRegisterForm(request.POST or None)
    if request.method=='POST' and form.is_valid():
        username=form.cleaned_data['username'];
        if User.objects.filter(username=username).exists(): form.add_error('username','Username already exists.')
        else:
            u=User.objects.create_user(username=username,email=form.cleaned_data['email'],password=form.cleaned_data['password']); c=Company.objects.create(name=form.cleaned_data['company_name'],slug=slugify(form.cleaned_data['company_name'])+'-'+str(u.id)); Membership.objects.create(company=c,user=u,role='OWNER'); _provision_company(c); login(request,u); messages.success(request,f'Welcome to {c.name}. Your private workspace is ready.'); return redirect('dashboard')
    return render(request,'predictor/auth.html',{'form':form,'mode':'register'})

def login_view(request):
    if request.user.is_authenticated:return redirect('dashboard')
    form=LoginForm(request.POST or None)
    if request.method=='POST' and form.is_valid():
        u=authenticate(request,username=form.cleaned_data['username'],password=form.cleaned_data['password'])
        if u: login(request,u); return redirect('dashboard')
        form.add_error(None,'Invalid username or password.')
    return render(request,'predictor/auth.html',{'form':form,'mode':'login'})

def logout_view(request): logout(request); return redirect('login')

def _provision_company(company):
    dd=Path(settings.COMPANY_DATA_DIR)/company.slug; dd.mkdir(parents=True,exist_ok=True); md=Path(settings.COMPANY_MODEL_DIR)/company.slug; md.mkdir(parents=True,exist_ok=True)
    for fn in ['users.csv','training_template.csv','test_template.csv']:
        src=Path(settings.DATA_DIR)/fn
        if src.exists() and not (dd/fn).exists():
            import shutil; shutil.copy2(src,dd/fn)
    Dataset.objects.get_or_create(company=company,name='Starter Audience Dataset',data_type='RAW',file_path=str(dd/'users.csv'),rows=len(pd.read_csv(dd/'users.csv')))
    Dataset.objects.get_or_create(company=company,name='Training Template',data_type='TRAIN',file_path=str(dd/'training_template.csv'),rows=len(pd.read_csv(dd/'training_template.csv')) if (dd/'training_template.csv').exists() else 0)
    Dataset.objects.get_or_create(company=company,name='Testing Template',data_type='TEST',file_path=str(dd/'test_template.csv'),rows=len(pd.read_csv(dd/'test_template.csv')) if (dd/'test_template.csv').exists() else 0)
    if not (dd/'audience_predictions.csv').exists():
        import shutil; shutil.copy2(Path(settings.DATA_DIR)/'audience_predictions.csv',dd/'audience_predictions.csv')

def dashboard(request):
    c = _company(request)
    if c is None:
        return redirect('login')

    _provision_company(c)
    d = _df(c).copy()
    rep = load(c)['report']

    # Defensive defaults so the dashboard never crashes when a dataset is
    # missing an optional analytics column.
    if 'churn_probability' not in d:
        d['churn_probability'] = 0.0
    if 'predicted_ltv' not in d:
        d['predicted_ltv'] = 0.0
    if 'engagement_health' not in d:
        d['engagement_health'] = (1 - d['churn_probability']).clip(0, 1) * 100
    if 'risk' not in d:
        d['risk'] = d['churn_probability'].apply(_risk_label)
    if 'segment' not in d:
        d['segment'] = 'Unclassified'
    if 'predicted_next_action' not in d:
        d['predicted_next_action'] = 'UNKNOWN'
    if 'language' not in d:
        d['language'] = 'Unknown'
    if 'region' not in d:
        d['region'] = 'Unknown'
    if 'subscription' not in d:
        d['subscription'] = 'Unknown'

    segment = d['segment'].value_counts().to_dict()
    risk = (
        d['risk'].value_counts()
        .reindex(['LOW', 'MODERATE', 'HIGH', 'CRITICAL'])
        .fillna(0)
        .astype(int)
        .to_dict()
    )
    actions = d['predicted_next_action'].value_counts().to_dict()

    # Retention queue: prioritize high-risk + high-value users. If that
    # combination is empty, fall back to the highest-risk users so the queue
    # always contains useful entries.
    ltv_cutoff = d['predicted_ltv'].quantile(0.55) if len(d) else 0
    priority = d[
        (d['churn_probability'] >= 0.50)
        & (d['predicted_ltv'] >= ltv_cutoff)
    ].nlargest(12, 'churn_probability')
    if priority.empty:
        priority = d.nlargest(12, 'churn_probability')

    # These analytics MUST be computed regardless of whether the retention
    # queue is empty. The old one-line `if` accidentally put these statements
    # inside the if-suite and caused `lang` to be unbound.
    lang = (
        d.groupby('language')
        .agg(
            churn=('churn_probability', 'mean'),
            health=('engagement_health', 'mean'),
            users=('user_id', 'count'),
        )
        .sort_values('churn', ascending=False)
        .round(4)
    )
    reg = (
        d.groupby('region')
        .agg(
            churn=('churn_probability', 'mean'),
            users=('user_id', 'count'),
            health=('engagement_health', 'mean'),
        )
        .sort_values('churn', ascending=False)
        .round(4)
    )
    sub = (
        d.groupby('subscription')
        .agg(
            churn=('churn_probability', 'mean'),
            ltv=('predicted_ltv', 'mean'),
            users=('user_id', 'count'),
        )
        .reset_index()
    )

    top_churn = d.nlargest(12, 'churn_probability')

    return render(
        request,
        'predictor/dashboard.html',
        {
            'company': c,
            'report': rep,
            'risk': risk,
            'segment': segment,
            'actions': actions,
            'priority': priority.to_dict('records'),
            'lang': lang.reset_index().to_dict('records'),
            'region': reg.reset_index().to_dict('records'),
            'subscription': sub.to_dict('records'),
            'total': len(d),
            'critical': int((d['risk'] == 'CRITICAL').sum()),
            'high': int((d['risk'] == 'HIGH').sum()),
            'churn_rate': float(d['churn_probability'].mean()),
            'avg_ltv': float(d['predicted_ltv'].mean()),
            'high_value_risk': len(priority),
            'health_avg': float(d['engagement_health'].mean()),
            'top_churn_users': top_churn[
                ['user_id', 'churn_probability', 'predicted_ltv', 'segment', 'risk']
            ].to_dict('records'),
            'charts': {
                'risk': _json({'labels': list(risk.keys()), 'data': list(risk.values())}),
                'segment': _json({'labels': list(segment.keys()), 'data': list(segment.values())}),
                'actions': _json({'labels': list(actions.keys()), 'data': list(actions.values())}),
                'lang': _json({'labels': lang.index.tolist(), 'churn': (lang.churn * 100).tolist()}),
                'region': _json({'labels': reg.index.tolist(), 'churn': (reg.churn * 100).tolist()}),
                'subscription': _json({
                    'labels': sub.subscription.tolist(),
                    'ltv': sub.ltv.round(2).tolist(),
                    'churn': (sub.churn * 100).round(2).tolist(),
                }),
                'health': _json({
                    'labels': ['0-20', '20-40', '40-60', '60-80', '80-100'],
                    'data': [
                        int(((d.engagement_health >= a) & (d.engagement_health < b)).sum())
                        for a, b in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]
                    ],
                }),
                'value_risk': _json({
                    'x': d.predicted_ltv.round(2).tolist(),
                    'y': (d.churn_probability * 100).round(2).tolist(),
                }),
            },
        },
    )

@login_required
def users(request):
    d=_df(_company(request)); q=request.GET.get('q','').strip(); risk=request.GET.get('risk',''); seg=request.GET.get('segment','');
    if q:d=d[d.user_id.astype(str).str.contains(q,case=False,na=False)]
    if risk:d=d[d.risk==risk]
    if seg:d=d[d.segment==seg]
    return render(request,'predictor/users.html',{'rows':d.sort_values('churn_probability',ascending=False).head(150).to_dict('records'),'segments':sorted(_df(_company(request)).segment.dropna().unique()),'query':q,'risk_filter':risk,'segment_filter':seg})

@login_required
def user_detail(request,user_id):
    d=_df(_company(request)); r=d[d.user_id.astype(str)==str(user_id)]
    if r.empty:return render(request,'predictor/not_found.html',status=404)
    row=r.iloc[0].to_dict(); vals={f:row[f] for f in FEATURES}; out=predict_one({'user_id':user_id,**vals},_company(request)); return render(request,'predictor/user_detail.html',{'row':row,'out':out,'behavior':_json({f:float(row[f]) for f in FEATURES}),'impacts':_json(out['impacts']),'profile':_json(out['profile']),'feature_names':FEATURES})

def _norm_key(k):
    return ''.join(ch for ch in str(k).lower().strip() if ch.isalnum())

def _extract_profile(upload):
    import io, re, json as _jsonlib
    name=upload.name.lower(); raw=upload.read(); upload.seek(0)
    aliases={
        _norm_key(f):f for f in FEATURES
    }
    extra={_norm_key('user id'):'user_id', _norm_key('userid'):'user_id', _norm_key('user'):'user_id'}
    aliases.update(extra)
    # Friendly aliases frequently used in exported CRM/analytics files.
    alias_words={
        'sessions':'sessions_7d','weekly_sessions':'sessions_7d','session_change':'session_change_7d',
        'listening_minutes':'listening_minutes_7d','weekly_listening_minutes':'listening_minutes_7d',
        'skip':'skip_rate','save':'save_rate','share':'share_rate','playlist':'playlist_rate',
        'search':'search_rate','discovery':'discovery_rate','artists':'unique_artists_7d','genres':'unique_genres_7d',
        'days_active':'days_active_14d','avg_session':'avg_session_minutes','subscription_age':'subscription_age_days',
        'support_tickets':'support_tickets_30d','night_share':'night_listening_share','completion':'completion_rate',
        'recommendation_clicks':'recommendation_click_rate','offline_downloads':'offline_downloads_30d',
        'playlists_created':'playlists_created_30d','social_interactions':'social_interactions_30d',
        'device_switches':'device_switch_rate','ad_skips':'ad_skip_rate','notification_fatigue_score':'notification_fatigue'
    }
    aliases.update({_norm_key(k):v for k,v in alias_words.items()})
    df=None; text=''
    try:
        if name.endswith('.csv'):
            df=pd.read_csv(io.BytesIO(raw))
        elif name.endswith(('.xlsx','.xls')):
            df=pd.read_excel(io.BytesIO(raw))
        elif name.endswith('.json'):
            obj=_jsonlib.loads(raw.decode('utf-8','ignore')); df=pd.DataFrame(obj if isinstance(obj,list) else [obj])
        elif name.endswith('.pdf'):
            from pypdf import PdfReader
            reader=PdfReader(io.BytesIO(raw)); text='\n'.join((page.extract_text() or '') for page in reader.pages)
        elif name.endswith(('.txt','.text')):
            text=raw.decode('utf-8','ignore')
        elif name.endswith('.docx'):
            from docx import Document
            doc=Document(io.BytesIO(raw)); text='\n'.join(x.text for x in doc.paragraphs)
        else:
            raise ValueError('Unsupported file type. Use CSV, XLSX, JSON, TXT, PDF or DOCX.')
    except Exception as e:
        raise ValueError(f'Could not read {upload.name}: {e}')
    if df is not None:
        if df.empty: raise ValueError('The uploaded file has no rows.')
        row=df.iloc[0].to_dict(); extracted={}
        for k,v in row.items():
            target=aliases.get(_norm_key(k))
            if target: extracted[target]=v
        missing=[f for f in FEATURES if f not in extracted or pd.isna(extracted[f])]
        if missing: raise ValueError('Missing model fields: '+', '.join(missing[:8])+(' ...' if len(missing)>8 else ''))
        extracted={k:float(v) if k!='user_id' else str(v) for k,v in extracted.items()}
        return extracted, f'Extracted 32 model features from row 1 of {upload.name}.'
    # Text/PDF/DOCX parser: accepts "feature: value" or "feature = value" lines.
    extracted={}
    for line in text.splitlines():
        m=re.match(r'\s*([A-Za-z0-9_ .()/-]+)\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*$', line)
        if not m: continue
        key=_norm_key(m.group(1)); target=aliases.get(key)
        if target: extracted[target]=float(m.group(2))
    uid=re.search(r'(?:user\s*id|userid|user)\s*[:=]\s*([A-Za-z0-9_.-]+)', text, re.I)
    if uid: extracted['user_id']=uid.group(1)
    missing=[f for f in FEATURES if f not in extracted]
    if missing: raise ValueError('Could not extract all 32 model fields from the document. Missing: '+', '.join(missing[:8])+(' ...' if len(missing)>8 else '')+'. Use feature: value lines or upload CSV/XLSX.')
    extracted.setdefault('user_id','FILE_USER_001')
    return extracted, f'Extracted 32 model features from {upload.name}.'

@login_required
def predict(request):
    out=None; extracted=None; extract_message=None
    form=PredictionForm(request.POST or None)
    file_form=PredictionFileForm(request.POST or None, request.FILES or None)
    if request.method=='POST':
        if 'profile_file' in request.FILES and file_form.is_valid():
            try:
                extracted,extract_message=_extract_profile(request.FILES['profile_file'])
                out=predict_one(extracted,_company(request))
            except Exception as e:
                messages.error(request,str(e))
        elif form.is_valid():
            out=predict_one(form.cleaned_data,_company(request))
    return render(request,'predictor/predict.html',{'form':form,'file_form':file_form,'out':out,'extracted':extracted,'extract_message':extract_message,'impacts':_json(out['impacts']) if out else '[]','profile':_json(out['profile']) if out else '[]','sensitivity':_json(out.get('sensitivity',[]) if out else [])})

@login_required
def model_page(request):
    c=_company(request); m=load(c); rep=m['report']; fi=rep.get('feature_importance',{}); return render(request,'predictor/model.html',{'report':rep,'runs':TrainingRun.objects.filter(company=c).order_by('-created_at')[:10],'fi':_json({'labels':list(fi.keys()),'data':list(fi.values())}),'cm':_json(rep.get('confusion_matrix',[[0,0],[0,0]])),'roc':_json(rep.get('roc_curve',{'fpr':[0,1],'tpr':[0,1]})),'pr':_json(rep.get('pr_curve',{'precision':[1,0],'recall':[0,1]}))})

@login_required
def train(request):
    c=_company(request); form=TrainingForm(request.POST or None,request.FILES or None)
    if request.method=='POST' and form.is_valid():
        try:
            f=request.FILES['dataset']; report=train_models(f,form.cleaned_data['mode'],c); TrainingRun.objects.create(company=c,source=f.name,mode=form.cleaned_data['mode'],rows=report['rows'],churn_accuracy=report['churn_accuracy'],churn_precision=report['churn_precision'],churn_recall=report['churn_recall'],churn_f1=report['churn_f1'],churn_auc=report['churn_roc_auc'],action_accuracy=report.get('action_accuracy'),ltv_r2=report.get('ltv_r2')); messages.success(request,f"Your company model was retrained. Accuracy {report['churn_accuracy']:.3f} • ROC-AUC {report['churn_roc_auc']:.3f}"); return redirect('model')
        except Exception as e:messages.error(request,str(e))
    return render(request,'predictor/train.html',{'form':form})

@login_required
def test_page(request):
    form=TestForm(request.POST or None,request.FILES or None); result=None
    if request.method=='POST' and form.is_valid():
        try: result=test_model(request.FILES['dataset'],_company(request))
        except Exception as e: messages.error(request,str(e))
    return render(request,'predictor/test.html',{'form':form,'result':result})

@login_required
def datasets(request):
    c=_company(request); _provision_company(c); return render(request,'predictor/datasets.html',{'datasets':Dataset.objects.filter(company=c).order_by('-created_at')})

@login_required
def dataset_download(request,dataset_id):
    ds=Dataset.objects.get(id=dataset_id,company=_company(request)); return FileResponse(open(ds.file_path,'rb'),as_attachment=True,filename=Path(ds.file_path).name)

@login_required
def quick_train(request):
    c=_company(request); form=LabeledExampleForm(request.POST or None)
    if request.method=='POST' and form.is_valid():
        dd=Path(settings.COMPANY_DATA_DIR)/c.slug; dd.mkdir(parents=True,exist_ok=True); path=dd/'quick_labeled_examples.csv'; d=form.cleaned_data.copy(); row={f:d[f] for f in FEATURES}; row['churn']=int(d['churn']); row['ltv']=d.get('ltv'); row['next_action']=d.get('next_action'); old=pd.read_csv(path) if path.exists() else pd.DataFrame(columns=FEATURES+['churn','ltv','next_action']); pd.concat([old,pd.DataFrame([row])],ignore_index=True).to_csv(path,index=False); messages.success(request,f'Saved labeled example to {c.name}.'); return redirect('quick_train')
    path=Path(settings.COMPANY_DATA_DIR)/c.slug/'quick_labeled_examples.csv'; count=len(pd.read_csv(path)) if path.exists() else 0; return render(request,'predictor/quick_train.html',{'form':form,'count':count})

@login_required
def feedback(request):
    c=_company(request); form=FeedbackForm(request.POST or None)
    if request.method=='POST' and form.is_valid():PredictionFeedback.objects.create(company=c,user_id=form.cleaned_data['user_id'],actual_churn=int(form.cleaned_data['actual_churn']) if form.cleaned_data['actual_churn'] else None,actual_action=form.cleaned_data['actual_action'],actual_ltv=form.cleaned_data['actual_ltv'],notes=form.cleaned_data['notes']); messages.success(request,'Outcome feedback saved to your company workspace.'); return redirect('feedback')
    return render(request,'predictor/feedback.html',{'form':form,'count':PredictionFeedback.objects.filter(company=c).count()})

@login_required
def train_template(request):
    c=_company(request); _provision_company(c); p=Path(settings.COMPANY_DATA_DIR)/c.slug/'training_template.csv'; return FileResponse(open(p,'rb'),as_attachment=True,filename=f'{c.slug}_training_template.csv')

@login_required
def test_template(request):
    c=_company(request); _provision_company(c); p=Path(settings.COMPANY_DATA_DIR)/c.slug/'test_template.csv'; return FileResponse(open(p,'rb'),as_attachment=True,filename=f'{c.slug}_test_template.csv')

def chatbot_answer(question,d):
    q=question.lower(); n=len(d); avg=d.churn_probability.mean()*100; high=d[d.churn_probability>=.5]; critical=d[d.risk=='CRITICAL']; top=d.nlargest(5,'churn_probability'); seg=d.segment.value_counts(); lang=d.groupby('language').churn_probability.mean().sort_values(ascending=False) if 'language' in d else pd.Series();
    if any(x in q for x in ['summary','summarise','summarize','overview']): return f"Audience summary: {n:,} users analyzed. Average predicted churn is {avg:.1f}%, average engagement health is {d.engagement_health.mean():.1f}/100, and average predicted LTV is ₹{d.predicted_ltv.mean():,.0f}. {len(high):,} users are high-risk or worse; {len(critical):,} are critical. The largest segment is {seg.index[0]} ({seg.iloc[0]:,} users)."
    if 'risk' in q or 'churn' in q:
        return f"There are {len(high):,} users at 50%+ predicted churn and {len(critical):,} critical users at 72%+. The highest-risk user is {top.iloc[0].user_id} at {top.iloc[0].churn_probability*100:.1f}% with predicted LTV ₹{top.iloc[0].predicted_ltv:,.0f}."
    if 'language' in q and not lang.empty:return f"Highest average churn by language: {lang.index[0]} at {lang.iloc[0]*100:.1f}%. Lowest: {lang.index[-1]} at {lang.iloc[-1]*100:.1f}%."
    if 'segment' in q:return 'Segments by size: '+', '.join(f'{k} ({v:,})' for k,v in seg.items())+'.'
    if 'action' in q:return 'Most likely next actions: '+', '.join(f'{k} ({v:,})' for k,v in d.predicted_next_action.value_counts().items())+'.'
    if 'ltv' in q or 'value' in q:return f"Average predicted LTV is ₹{d.predicted_ltv.mean():,.0f}. The top 10% average ₹{d.predicted_ltv.quantile(.9):,.0f}+. Focus retention on users who combine high LTV with high churn probability."
    if 'recommend' in q or 'retention' in q:return f"Retention priority: target the {len(high):,} high-risk users, starting with high-LTV users. For skip-heavy users, improve personalization; for low-session users, use re-engagement; for low-discovery users, strengthen discovery surfaces."
    return "I can summarize your audience, explain churn risk, compare languages/regions/subscriptions, inspect segments, discuss LTV, next actions, or suggest retention priorities. Try: 'Summarize everything' or 'Which users should we retain?'"

@login_required
def chatbot(request):
    c=_company(request); d=_df(c); form=ChatForm(request.POST or None); answer=None
    if request.method=='POST' and form.is_valid():answer=chatbot_answer(form.cleaned_data['question'],d)
    return render(request,'predictor/chatbot.html',{'form':form,'answer':answer,'company':c})

# JWT API
@api_view(['POST'])
@permission_classes([])
def api_register(request):
    company_name=request.data.get('company_name'); username=request.data.get('username'); email=request.data.get('email',''); password=request.data.get('password')
    if not all([company_name,username,password]): return Response({'error':'company_name, username and password are required'},status=400)
    if User.objects.filter(username=username).exists(): return Response({'error':'username already exists'},status=400)
    u=User.objects.create_user(username=username,email=email,password=password); c=Company.objects.create(name=company_name,slug=slugify(company_name)+'-'+str(u.id)); Membership.objects.create(company=c,user=u,role='OWNER'); _provision_company(c); return Response({'message':'company registered','company':c.name,'username':u.username})

@api_view(['GET'])
def api_me(request):
    c=_company(request); return Response({'username':request.user.username,'email':request.user.email,'company':c.name if c else None,'role':request.user.membership.role if hasattr(request.user,'membership') else None})

@api_view(['POST'])
def api_predict(request):
    try:return Response(predict_one(request.data,_company(request)))
    except Exception as e:return Response({'error':str(e)},status=400)
