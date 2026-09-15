from django import forms
FEATURES=['sessions_7d','session_change_7d','listening_minutes_7d','listening_change_7d','skip_rate','save_rate','share_rate','playlist_rate','search_rate','discovery_rate','unique_artists_7d','unique_genres_7d','days_active_14d','avg_session_minutes','subscription_age_days','support_tickets_30d','night_listening_share','completion_rate','avg_daily_sessions','weekend_usage_share','session_gap_hours','artist_concentration','genre_concentration','notification_open_rate','recommendation_click_rate','offline_downloads_30d','playlists_created_30d','social_interactions_30d','device_switch_rate','ad_skip_rate','notification_fatigue','content_diversity']
RATE_FEATURES={'skip_rate','save_rate','share_rate','playlist_rate','search_rate','discovery_rate','night_listening_share','completion_rate'}
class PredictionFileForm(forms.Form):
    profile_file=forms.FileField(label='Upload user profile (CSV, XLSX, JSON, TXT, PDF or DOCX)', required=True)

class PredictionForm(forms.Form):
    user_id=forms.CharField(initial='NEW_USER_001',label='User ID')
    for f in FEATURES:
        kwargs={'required':True,'label':f.replace('_',' ').title(),'widget':forms.NumberInput(attrs={'step':'0.01'})}
        if f not in {'session_change_7d','listening_change_7d'}: kwargs['min_value']=0
        locals()[f]=forms.FloatField(**kwargs)
    def clean(self):
        d=super().clean()
        for f in RATE_FEATURES:
            if d.get(f) is not None and d[f]>1:d[f]/=100
        return d
class TrainingForm(forms.Form):
    dataset=forms.FileField(label='Labeled CSV dataset')
    mode=forms.ChoiceField(choices=[('combine','Combine with company training data'),('replace','Replace company training data')],initial='combine',widget=forms.RadioSelect)
class TestForm(forms.Form):
    dataset=forms.FileField(label='Labeled test CSV')
class FeedbackForm(forms.Form):
    user_id=forms.CharField(label='User ID'); actual_churn=forms.ChoiceField(choices=[('','Unknown'),('0','Stayed active'),('1','Churned')],required=False); actual_action=forms.CharField(required=False); actual_ltv=forms.FloatField(required=False,min_value=0); notes=forms.CharField(required=False,widget=forms.Textarea(attrs={'rows':3}))
class LabeledExampleForm(forms.Form):
    user_id=forms.CharField(initial='LABELED_001',label='Example/User ID')
    for f in FEATURES:
        kwargs={'required':True,'label':f.replace('_',' ').title(),'widget':forms.NumberInput(attrs={'step':'0.01'})}
        if f not in {'session_change_7d','listening_change_7d'}: kwargs['min_value']=0
        locals()[f]=forms.FloatField(**kwargs)
    churn=forms.ChoiceField(choices=[('1','Churned'),('0','Stayed active')],label='Observed churn outcome')
    ltv=forms.FloatField(required=False,min_value=0,label='Observed LTV (optional)')
    next_action=forms.ChoiceField(required=False,choices=[('','Unknown'),('PLAY','PLAY'),('SEARCH','SEARCH'),('SKIP','SKIP'),('SAVE','SAVE'),('SHARE','SHARE'),('CREATE_PLAYLIST','CREATE_PLAYLIST'),('LEAVE_APP','LEAVE_APP')],label='Observed next action')
    def clean(self):
        d=super().clean()
        for f in RATE_FEATURES:
            if d.get(f) is not None and d[f]>1:d[f]/=100
        return d
class CompanyRegisterForm(forms.Form):
    company_name=forms.CharField(max_length=160,label='Company name')
    username=forms.CharField(max_length=150)
    email=forms.EmailField()
    password=forms.CharField(widget=forms.PasswordInput)
    password2=forms.CharField(widget=forms.PasswordInput,label='Confirm password')
    def clean(self):
        d=super().clean()
        if d.get('password')!=d.get('password2'): raise forms.ValidationError('Passwords do not match.')
        return d
class LoginForm(forms.Form):
    username=forms.CharField(); password=forms.CharField(widget=forms.PasswordInput)
class ChatForm(forms.Form):
    question=forms.CharField(widget=forms.TextInput(attrs={'placeholder':'Ask: Which users are most at risk?'}))
