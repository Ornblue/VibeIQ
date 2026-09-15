from django.contrib import admin
from .models import Company,Membership,UserPrediction,TrainingRun,PredictionFeedback,Dataset
for m in [Company,Membership,UserPrediction,TrainingRun,PredictionFeedback,Dataset]: admin.site.register(m)
