# VibeIQ VibeIQ — ULTIMATE

A multi-company, end-to-end data science platform inspired by streaming-platform audience analytics. It focuses on listener behavior rather than songs.

## Included
- Multi-tenant company registration and isolated company datasets/model artifacts
- Django web authentication + JWT API authentication
- Churn prediction with Random Forest
- Engagement Health Score
- K-Means behavioral segmentation
- Next-action prediction
- LTV prediction
- Local counterfactual explainability
- 8+ dashboard charts and retention queue
- AI Audience Copilot: company-scoped natural-language analytics summaries
- Company-specific training/retraining with Combine or Replace mode
- Company-specific model testing with Accuracy, Precision, Recall, F1, ROC-AUC and confusion matrix
- Dataset Hub: starter audience, training template and test template for every company
- Label & Learn feedback loop
- Model history and metrics
- Seed demo command
- Standalone ML notebook and source data

## Important data note
All included data is synthetic demonstration data. It is not real VibeIQ customer data and the model metrics are not claims about VibeIQ.

## Mac setup
```bash
cd ~/Desktop
unzip VibeIQ_Audience_Intelligence_Django_ULTIMATE.zip
cd VibeIQ_Audience_Intelligence_Django_ULTIMATE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python manage.py migrate
python manage.py seed_demo
python manage.py check
python manage.py runserver
```
Open http://127.0.0.1:8000/

Demo login: demo_admin / demo1234

## JWT API
Get token:
POST /api/auth/token/
```json
{"username":"demo_admin","password":"demo1234"}
```
Refresh token:
POST /api/auth/token/refresh/

Register company:
POST /api/auth/register/
```json
{"company_name":"Acme Streaming","username":"acme_admin","email":"admin@acme.local","password":"strongpassword"}
```

Authenticated API:
GET /api/me/
POST /api/predict/
Use header: Authorization: Bearer <access_token>

## Training CSV
Required columns:
- all 18 behavioral features in data/training_template.csv
- churn (0/1)
Optional:
- ltv
- next_action
Minimum 50 rows and both churn classes.

## Testing CSV
Use data/test_template.csv or your own labeled dataset. It must contain the 18 features plus churn.

## Multi-company isolation
Each registered company gets:
- data/companies/<company-slug>/users.csv
- training_template.csv
- test_template.csv
- audience_predictions.csv
- active_training_data.csv after training
- company_models/<company-slug>/ model artifacts

A company's uploaded data is used only for that company's model. The web UI filters all predictions, feedback and training history by the logged-in company.

## AI Copilot
The included Copilot is a local analytics assistant: it computes answers from the current company's dataset without requiring an external API key. It can summarize the audience, churn/risk, segments, LTV, actions, language patterns and retention priorities.

## SQLite seed safety

`python manage.py seed_demo` now checks for all required predictor tables. If a stale SQLite migration state says the app is migrated while the tables are missing, it automatically resets and reapplies only the `predictor` migrations before seeding. This does not delete model artifacts or datasets.
