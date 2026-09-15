from django.conf import settings
from django.db import models

class Company(models.Model):
    name = models.CharField(max_length=160)
    slug = models.SlugField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self): return self.name

class Membership(models.Model):
    ROLE_CHOICES=[('OWNER','Owner'),('ANALYST','Analyst')]
    company=models.ForeignKey(Company,on_delete=models.CASCADE,related_name='memberships')
    user=models.OneToOneField(settings.AUTH_USER_MODEL,on_delete=models.CASCADE,related_name='membership')
    role=models.CharField(max_length=20,choices=ROLE_CHOICES,default='OWNER')
    def __str__(self): return f'{self.user} @ {self.company}'

class UserPrediction(models.Model):
    company=models.ForeignKey(Company,on_delete=models.CASCADE,related_name='predictions')
    user_id=models.CharField(max_length=64)
    segment=models.CharField(max_length=80,blank=True)
    churn_probability=models.FloatField(default=0)
    engagement_health=models.FloatField(default=0)
    predicted_ltv=models.FloatField(default=0)
    next_action=models.CharField(max_length=40,blank=True)
    risk=models.CharField(max_length=20,blank=True)
    recommendation=models.CharField(max_length=300,blank=True)
    updated_at=models.DateTimeField(auto_now=True)
    class Meta:
        constraints=[models.UniqueConstraint(fields=['company','user_id'],name='uniq_company_userprediction')]
    def __str__(self): return f'{self.company.slug}:{self.user_id}'

class TrainingRun(models.Model):
    company=models.ForeignKey(Company,on_delete=models.CASCADE,related_name='training_runs')
    created_at=models.DateTimeField(auto_now_add=True)
    source=models.CharField(max_length=255)
    mode=models.CharField(max_length=30)
    rows=models.IntegerField()
    churn_accuracy=models.FloatField(null=True)
    churn_precision=models.FloatField(null=True)
    churn_recall=models.FloatField(null=True)
    churn_f1=models.FloatField(null=True)
    churn_auc=models.FloatField(null=True)
    action_accuracy=models.FloatField(null=True)
    ltv_r2=models.FloatField(null=True)
    segments=models.IntegerField(default=5)
    def __str__(self): return f'{self.company.name} {self.created_at:%Y-%m-%d %H:%M} {self.mode}'

class PredictionFeedback(models.Model):
    company=models.ForeignKey(Company,on_delete=models.CASCADE,related_name='feedback')
    user_id=models.CharField(max_length=64)
    actual_churn=models.IntegerField(null=True,blank=True)
    actual_action=models.CharField(max_length=40,blank=True)
    actual_ltv=models.FloatField(null=True,blank=True)
    notes=models.TextField(blank=True)
    created_at=models.DateTimeField(auto_now_add=True)

class Dataset(models.Model):
    DATA_TYPES=[('TRAIN','Training'),('TEST','Testing'),('RAW','Raw audience')]
    company=models.ForeignKey(Company,on_delete=models.CASCADE,related_name='datasets')
    name=models.CharField(max_length=255)
    data_type=models.CharField(max_length=10,choices=DATA_TYPES)
    file_path=models.CharField(max_length=500)
    rows=models.IntegerField(default=0)
    created_at=models.DateTimeField(auto_now_add=True)
    def __str__(self): return f'{self.company.name} / {self.name}'
