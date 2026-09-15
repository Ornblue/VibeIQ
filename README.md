# VibeIQ — Audience Intelligence & Customer Behavior Platform

> **VibeIQ is an end-to-end, multi-company audience intelligence platform that converts behavioral data into churn-risk predictions, customer lifetime value estimates, behavioral segments, next-action predictions, engagement health scores, retention priorities, explainable recommendations, and natural-language analytics.**

VibeIQ is designed as a portfolio-grade **machine learning (ML) + Django web application** rather than a standalone notebook. It demonstrates the complete path from raw behavioral data to trained models, predictions, explainability, dashboards, feedback, retraining, testing, APIs, and a multi-tenant production-style application structure.

---

## Table of Contents

1. [What is VibeIQ?](#what-is-vibeiq)
2. [The Problem](#the-problem)
3. [What VibeIQ Predicts](#what-vibeiq-predicts)
4. [How the System Works](#how-the-system-works)
5. [Architecture](#architecture)
6. [Machine Learning Models](#machine-learning-models)
7. [Dataset](#dataset)
8. [Complete Feature Dictionary](#complete-feature-dictionary)
9. [Target Variables](#target-variables)
10. [Risk Levels](#risk-levels)
11. [Behavioral Segmentation](#behavioral-segmentation)
12. [Engagement Health Score](#engagement-health-score)
13. [Explainability](#explainability)
14. [Retention Intelligence](#retention-intelligence)
15. [AI Audience Copilot](#ai-audience-copilot)
16. [File-Based Prediction](#file-based-prediction)
17. [Training and Retraining](#training-and-retraining)
18. [Model Testing](#model-testing)
19. [Label & Learn Feedback Loop](#label--learn-feedback-loop)
20. [Multi-Company Architecture](#multi-company-architecture)
21. [Authentication and API](#authentication-and-api)
22. [Dashboard and Application Pages](#dashboard-and-application-pages)
23. [Current Demo Dataset](#current-demo-dataset)
24. [Current Model Performance](#current-model-performance)
25. [Project Structure](#project-structure)
26. [Installation on macOS](#installation-on-macos)
27. [Using the Application](#using-the-application)
28. [Using Your Own Data](#using-your-own-data)
29. [Training CSV Requirements](#training-csv-requirements)
30. [Testing CSV Requirements](#testing-csv-requirements)
31. [Supported File Formats](#supported-file-formats)
32. [Technology Stack](#technology-stack)
33. [Abbreviation Glossary](#abbreviation-glossary)
34. [Important Limitations](#important-limitations)
35. [Future Improvements](#future-improvements)
36. [License / Academic Use](#license--academic-use)

---

# What is VibeIQ?

VibeIQ is an **audience intelligence** system for organizations that want to understand users beyond a single prediction.

Instead of answering only:

> "Will this customer churn?"

VibeIQ attempts to answer a broader set of business questions:

- Which users are likely to churn?
- How valuable is each user likely to be?
- What behavioral segment does the user belong to?
- What action is the user most likely to take next?
- How healthy is the user's current engagement?
- Which behavioral signals are increasing or decreasing predicted churn?
- Which users should the retention team prioritize?
- What intervention might be appropriate?
- Can an analyst upload a user profile instead of manually entering 32 values?
- Can the organization retrain its models using its own labeled data?
- Can model performance be evaluated on a separate test dataset?
- Can actual outcomes be fed back into the system for future learning?
- Can multiple companies use the same application while keeping their data and models isolated?

That makes VibeIQ a combination of:

**Data Engineering → Machine Learning → Explainable AI → Analytics → Web Application → API → Feedback → Retraining**

---

# The Problem

A behavioral platform can collect thousands of user interactions, but raw events alone are difficult to act upon.

For example, a user may have:

- fewer sessions than last week,
- increasing session gaps,
- a high skip rate,
- low discovery behavior,
- low recommendation interaction,
- low completion rate,
- high notification fatigue,
- and a declining listening pattern.

Individually, each metric provides only partial information. VibeIQ combines these signals into a **32-dimensional behavioral profile** and uses machine learning to generate multiple forms of intelligence.

The central idea is:

```text
Raw / Aggregated User Behavior
              ↓
       32 Behavioral Features
              ↓
      Machine Learning Models
              ↓
 ┌────────────┼──────────────┐
 ↓            ↓              ↓
Churn       LTV          Next Action
Risk       Estimate       Prediction
 ↓            ↓              ↓
 └────────────┼──────────────┘
              ↓
       K-Means Segmentation
              ↓
     Engagement Health Score
              ↓
 Explainability + Recommendations
              ↓
      Retention Prioritization
              ↓
      Human Feedback / Labels
              ↓
          Retraining
```

---

# What VibeIQ Predicts

## 1. Churn Probability

**Churn** means that a user is likely to stop being active or leave the platform.

The churn model produces a probability between `0` and `1`.

For example:

```text
0.08 → approximately 8% predicted churn probability
0.47 → approximately 47% predicted churn probability
0.81 → approximately 81% predicted churn probability
```

The application displays this as a percentage.

---

## 2. Predicted LTV

**LTV = Lifetime Value.**

LTV estimates the expected monetary value associated with a user over their relationship with the platform.

The project predicts LTV as a regression problem rather than a classification problem.

The shipped model is trained using a logarithmic transformation of LTV and converted back using the exponential transformation during prediction. This helps the model handle the highly skewed distribution that monetary values often have.

---

## 3. Next Action

The next-action model predicts the behavioral action a user is most likely to take.

The current action classes include:

- `PLAY`
- `SEARCH`
- `SKIP`
- `SAVE`
- `SHARE`
- `CREATE_PLAYLIST`
- `LEAVE_APP`

This turns VibeIQ from a passive risk dashboard into a system that can also reason about likely immediate behavior.

---

## 4. Behavioral Segment

VibeIQ uses **K-Means clustering** to group users with similar behavioral profiles.

The shipped demo contains six named segments:

1. **Daily Loyalist** — consistently active users with strong recurring engagement.
2. **Explorer** — users with strong discovery and varied content behavior.
3. **Binge Listener** — users with high listening intensity and long/strong sessions.
4. **Community Builder** — users with strong social, sharing, saving, playlist, or collaborative behavior.
5. **Drifting User** — users showing weakening or inconsistent engagement.
6. **Ghost / Low Activity** — users with very limited or weak recent activity.

The segment names are business-friendly interpretations of unsupervised clusters. They are not directly supplied as labels to K-Means.

---

# How the System Works

VibeIQ follows this high-level pipeline:

### Step 1 — Data generation / ingestion

The demo includes synthetic user and event data. A real deployment could replace this with production data sources.

### Step 2 — Feature representation

Raw behavior is represented using 32 engineered numerical features.

### Step 3 — Model inference

Four model components generate:

- churn probability,
- predicted LTV,
- predicted next action,
- behavioral segment.

### Step 4 — Business interpretation

The raw model outputs are converted into:

- risk levels,
- engagement health,
- retention recommendations,
- explainability signals,
- retention queues.

### Step 5 — Visualization

Django renders dashboards, charts, tables, user profiles, model metrics, and explanations.

### Step 6 — Learning loop

Users can upload labeled data, test models, and provide actual outcomes through **Label & Learn**.

---

# Architecture

```text
                        ┌─────────────────────┐
                        │      Browser        │
                        │   VibeIQ Web UI     │
                        └──────────┬──────────┘
                                   │ HTTP
                                   ↓
                        ┌─────────────────────┐
                        │       Django        │
                        │ Views + Forms + URL │
                        └──────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ↓                    ↓                    ↓
       ┌─────────────┐      ┌──────────────┐      ┌──────────────┐
       │ Model       │      │ Training     │      │ Database     │
       │ Service     │      │ Service      │      │ SQLite       │
       └──────┬──────┘      └──────┬───────┘      └──────────────┘
              │                    │
              ↓                    ↓
       ┌─────────────┐      ┌──────────────┐
       │ .pkl Model  │      │ Training CSV │
       │ Artifacts   │      │ / Test CSV   │
       └─────────────┘      └──────────────┘
              │
              ↓
       ┌───────────────────────────────────┐
       │ Churn + LTV + Action + Segments  │
       └───────────────────────────────────┘
```

The project also exposes a **REST API** for programmatic access.

---

# Machine Learning Models

VibeIQ uses multiple machine learning approaches because the outputs represent different mathematical problems.

## Random Forest Classifier — Churn

A **Random Forest** is an ensemble machine learning algorithm that combines many decision trees.

A **decision tree** repeatedly splits data using feature conditions to reach a prediction.

A Random Forest builds many trees and combines their predictions, usually improving robustness compared with relying on a single tree.

The churn model is a:

```text
RandomForestClassifier
```

The shipped model uses:

- 50 trees (`n_estimators=50`)
- maximum tree depth of 10
- minimum 5 samples per leaf
- balanced class weighting
- fixed random seed of 42

The training service can train a larger 300-tree model when a company retrains its models.

### Why Random Forest for churn?

Churn is a **binary classification** problem:

```text
0 → Stayed active
1 → Churned
```

Random Forest works well with mixed nonlinear behavioral relationships and does not require feature scaling for tree-based prediction.

---

## Random Forest Regressor — LTV

The LTV model is a:

```text
RandomForestRegressor
```

A regression model predicts a continuous numerical value instead of a class.

Example:

```text
Predicted LTV = ₹842.50
```

The project models `log(1 + LTV)` during training and transforms the prediction back with `expm1`.

### Why regression?

LTV is numerical and can take many values, so classification would unnecessarily divide it into arbitrary categories.

---

## Random Forest Classifier — Next Action

The next-action model is another Random Forest classifier, but its target has multiple categories.

This is a **multiclass classification** problem.

Example:

```text
Input behavioral profile
        ↓
Random Forest
        ↓
CREATE_PLAYLIST
```

---

## StandardScaler + K-Means — Segmentation

Segmentation is different from churn prediction because there is no known target label.

The project therefore uses **unsupervised learning**.

### StandardScaler

`StandardScaler` transforms features so that numerical variables are placed on a comparable scale, approximately centered around zero with unit variance.

This is important for K-Means because distance calculations can otherwise be dominated by large-unit variables.

### K-Means

**K-Means** is an unsupervised clustering algorithm.

It attempts to divide users into `K` groups by minimizing the distance between users and their assigned cluster center.

The shipped artifacts use:

```text
K = 6
```

The resulting numeric cluster IDs are then mapped to human-readable segment names.

---

# Dataset

The project deliberately uses **synthetic data** for demonstration.

Synthetic means the values were generated programmatically rather than copied from a real company's private customer database.

This makes the repository safe to demonstrate without exposing confidential customer information.

## Main dataset files

### `data/users.csv`

Contains the primary user-level behavioral dataset.

Current shipped size:

- **15,000 users**
- **39 columns**
- 32 behavioral model features
- metadata columns
- observed labels for training/demo purposes

The 39 columns consist of:

```text
user_id
region
language
subscription
32 behavioral features
churn
ltv
next_action
```

---

### `data/streaming_events.csv`

Contains event-level synthetic behavior.

Current shipped size:

- **50,000 events**
- 5 columns

Columns:

```text
user_id
timestamp
event_type
language
duration_seconds
```

This file demonstrates what lower-level interaction data can look like before aggregation into user-level behavioral features.

---

### `data/audience_predictions.csv`

This is the scored audience table.

It combines the original user attributes/features with model-generated outputs such as:

- segment ID,
- segment name,
- churn probability,
- predicted LTV,
- predicted next action,
- engagement health,
- risk level,
- recommendation.

Current shipped size:

- **15,000 users**
- **47 columns**

---

### `data/training_template.csv`

A template showing the structure required for model retraining.

It contains:

- 32 behavioral features
- `churn`
- `ltv`
- `next_action`

---

### `data/test_template.csv`

A template for evaluating the churn model on labeled test data.

It contains:

- 32 behavioral features
- `churn`
- optional LTV/action columns can be present, but churn is required for the current testing workflow.

---

# Complete Feature Dictionary

VibeIQ uses exactly **32 behavioral features** for the core models.

## 1. `sessions_7d`

Number of user sessions during the previous seven days.

Higher values generally indicate stronger recent activity.

## 2. `session_change_7d`

Recent change in session activity.

Positive values indicate increasing session activity; negative values indicate decline.

## 3. `listening_minutes_7d`

Total listening time during the previous seven days.

This captures engagement intensity.

## 4. `listening_change_7d`

Change in listening time relative to the user's earlier baseline.

A negative value can indicate declining engagement.

## 5. `skip_rate`

Fraction of content starts that resulted in a skip.

Example:

```text
0.60 = 60% skip rate
```

High skip behavior can indicate poor content matching or dissatisfaction.

## 6. `save_rate`

Fraction of relevant interactions that resulted in a save.

Saving content is treated as a positive engagement signal.

## 7. `share_rate`

Fraction of relevant interactions resulting in sharing.

Sharing is a strong social engagement signal.

## 8. `playlist_rate`

Rate at which users add or organize content into playlists.

It reflects deeper intentional engagement.

## 9. `search_rate`

Rate of interactions involving search.

It helps distinguish users who actively seek content from users who primarily consume recommendations.

## 10. `discovery_rate`

Rate of exploratory/discovery interactions.

Higher discovery behavior generally indicates willingness to explore new content.

## 11. `unique_artists_7d`

Number of unique artists interacted with during seven days.

Higher values generally indicate broader artist exploration.

## 12. `unique_genres_7d`

Number of unique genres interacted with during seven days.

This captures breadth of content preference.

## 13. `days_active_14d`

Number of distinct active days during the previous 14 days.

It captures consistency rather than just total volume.

## 14. `avg_session_minutes`

Average session duration in minutes.

Longer sessions can indicate deeper engagement, although the model learns the relationship from data rather than assuming it is always positive.

## 15. `subscription_age_days`

Number of days since the user's subscription started.

This represents customer tenure.

## 16. `support_tickets_30d`

Number of support tickets created during the previous 30 days.

Higher support activity can indicate friction or unresolved problems.

## 17. `night_listening_share`

Fraction of listening activity occurring during nighttime.

This describes usage timing and routine.

## 18. `completion_rate`

Fraction of started content that the user completes.

Higher completion generally represents stronger content satisfaction or engagement.

## 19. `avg_daily_sessions`

Average number of sessions per active day.

This differentiates users with occasional activity from highly recurring users.

## 20. `weekend_usage_share`

Share of activity occurring on weekends.

It captures weekly usage patterns.

## 21. `session_gap_hours`

Average gap between sessions in hours.

Larger gaps can indicate weaker usage frequency.

## 22. `artist_concentration`

How concentrated a user's behavior is around a small number of artists.

A higher value means behavior is more concentrated rather than distributed.

## 23. `genre_concentration`

How concentrated behavior is around a small number of genres.

It is another measure of behavioral breadth versus narrowness.

## 24. `notification_open_rate`

Fraction of notifications opened by the user.

This indicates responsiveness to re-engagement or communication attempts.

## 25. `recommendation_click_rate`

Fraction of recommendation opportunities that receive a click.

It captures how strongly the user responds to personalized recommendations.

## 26. `offline_downloads_30d`

Number of offline downloads during the previous 30 days.

This can indicate strong intent to consume content repeatedly.

## 27. `playlists_created_30d`

Number of playlists created during the previous 30 days.

This represents active content organization and creation behavior.

## 28. `social_interactions_30d`

Number of social interactions during the previous 30 days.

This includes behaviors such as sharing or other social actions represented by the dataset.

## 29. `device_switch_rate`

Rate at which the user changes devices during usage.

It captures cross-device behavior.

## 30. `ad_skip_rate`

Fraction of advertisements skipped when skipping is available.

This represents interaction with monetization-related content.

## 31. `notification_fatigue`

A behavioral measure representing repeated exposure or reduced responsiveness to notifications.

Higher values can indicate communication fatigue.

## 32. `content_diversity`

A measure of how broadly the user consumes different content.

Higher values indicate a more diverse behavioral profile.

---

# Target Variables

The training dataset contains three main target variables.

## `churn`

Binary target:

```text
0 → stayed active
1 → churned
```

This is used by the churn classifier.

## `ltv`

Numerical target representing Lifetime Value.

This is used by the regression model.

## `next_action`

Categorical target representing the observed next user action.

This is used by the multiclass classifier.

---

# Risk Levels

The churn probability is converted into four business-facing risk categories.

| Churn Probability | Risk |
|---:|---|
| `< 25%` | LOW |
| `25% – < 50%` | MODERATE |
| `50% – < 72%` | HIGH |
| `>= 72%` | CRITICAL |

These thresholds are application-level business rules applied **after** the model predicts a probability.

They are not the same thing as the model's training labels.

For example, the model is trained on:

```text
churn = 0 or 1
```

It is **not** trained on:

```text
LOW / MODERATE / HIGH / CRITICAL
```

This distinction is important.

---

# Behavioral Segmentation

Segmentation is performed independently from churn classification.

The system first scales the 32-dimensional feature space using `StandardScaler`, then applies K-Means clustering.

Conceptually:

```text
User A → [32 behavioral values]
User B → [32 behavioral values]
User C → [32 behavioral values]
                 ↓
          StandardScaler
                 ↓
             K-Means
                 ↓
          Cluster ID 0–5
                 ↓
       Business-friendly name
```

The six shipped segments are:

| Segment | General behavioral interpretation |
|---|---|
| Daily Loyalist | Consistent, recurring engagement |
| Explorer | High exploration and content discovery |
| Binge Listener | High-intensity listening behavior |
| Community Builder | Strong social/creation behavior |
| Drifting User | Declining or inconsistent engagement |
| Ghost / Low Activity | Very low recent activity |

These labels are interpretations of cluster profiles, not supervised ground-truth categories.

---

# Engagement Health Score

VibeIQ exposes an easy-to-read engagement score from `0` to `100`.

The current implementation derives it from churn probability:

```text
Engagement Health = 100 × (1 − churn probability)
```

Example:

```text
Churn probability = 0.20
Health = 80/100
```

and:

```text
Churn probability = 0.85
Health = 15/100
```

This is intentionally a **derived business indicator**, not an independently trained machine learning model.

---

# Explainability

VibeIQ does not stop at displaying a probability.

The Predict page provides several views designed to answer:

> "Why did the model produce this result?"

## Signed churn drivers

For each important feature, VibeIQ replaces the user's value with the audience median and measures how the churn probability changes.

If replacing a feature with the median lowers predicted churn, the user's original value was contributing to higher predicted risk.

If the probability increases after replacing it with the median, the original value was comparatively protective.

This produces a **local counterfactual sensitivity analysis**.

It is not a formal causal claim.

---

## Behavioral peer profile

The application also calculates the user's percentile-like strength relative to the current audience for selected behavioral features.

This supports visual comparison between:

```text
Current user behavior
vs.
Observed audience distribution
```

---

## Counterfactual sensitivity

The UI compares:

```text
Current churn probability
vs.
Churn probability after replacing one feature with its audience median
```

This answers questions such as:

> "If this user's skip rate looked more like the median user, how would the model's prediction change?"

Again, this is a model sensitivity analysis rather than proof that changing the feature will cause the predicted outcome.

---

## Protective and risk-increasing signals

The system separates the largest positive and negative feature impacts into two lists so analysts can quickly identify:

- signals increasing predicted churn,
- signals associated with lower predicted churn.

---

# Retention Intelligence

The dashboard includes a retention queue designed to focus attention on users who are both:

1. sufficiently likely to churn, and
2. sufficiently valuable to prioritize.

The main priority logic considers:

```text
churn probability >= 0.50
AND
predicted LTV >= audience LTV 55th percentile
```

The queue then prioritizes users by churn probability.

If no user satisfies the combined condition, the system falls back to the highest-risk users so that the retention queue does not become empty simply because of a threshold intersection.

This is a **business prioritization rule**, not another machine learning model.

---

# AI Audience Copilot

VibeIQ includes a local **AI Audience Copilot**.

Unlike a cloud Large Language Model (LLM) integration, the included Copilot does not require an external API key.

It works by computing analytics from the current company's data and responding to common natural-language analytics questions.

Examples:

```text
Which users are most at risk?
```

```text
Give me an audience summary.
```

```text
Which segment is largest?
```

```text
What retention actions should we consider?
```

```text
Which users have high LTV?
```

The Copilot is intentionally company-scoped so that its calculations operate on the logged-in company's dataset.

---

# File-Based Prediction

Manual entry of 32 features is inconvenient.

VibeIQ therefore supports uploading a user profile and extracting the feature values automatically.

Supported formats:

- CSV — Comma-Separated Values
- XLSX — Microsoft Excel Open XML Spreadsheet
- XLS — legacy Microsoft Excel format where supported by the installed reader
- JSON — JavaScript Object Notation
- TXT — plain text
- PDF — Portable Document Format
- DOCX — Microsoft Word Open XML document

The extractor supports common aliases.

For example, a file may use:

```text
sessions
skip
save
share
completion
recommendation_clicks
offline_downloads
```

and VibeIQ can map these to its internal feature names where an alias is supported.

Text documents can also use simple structures such as:

```text
sessions_7d: 14
skip_rate: 0.31
completion_rate: 0.72
content_diversity: 0.64
```

The application validates the extracted feature set before prediction.

---

# Training and Retraining

VibeIQ supports company-specific model retraining.

A company can upload labeled CSV data and choose:

### Combine

Combine the new labeled data with the company's existing training data.

### Replace

Replace the company's current training dataset with the uploaded labeled dataset.

The training service then:

1. validates required features,
2. validates the churn target,
3. checks that at least 50 usable rows exist,
4. checks that both churn classes are present,
5. trains the churn model,
6. trains the LTV model when LTV labels are available,
7. trains the next-action model when action labels are available,
8. scales the behavioral features,
9. trains the K-Means segment model,
10. saves the new artifacts,
11. stores the active training dataset,
12. writes a model report,
13. re-scores the company's audience,
14. records a training run in the database.

Previous company model files are backed up before replacement.

---

# Model Testing

The **Test Model** page allows a company to upload a separate labeled test CSV.

The current churn testing workflow reports:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

This is important because training metrics alone do not tell us whether a model generalizes to unseen data.

---

# Label & Learn Feedback Loop

The **Label & Learn** functionality lets users record observed outcomes after predictions have been made.

Feedback can include:

- actual churn outcome,
- actual next action,
- actual LTV,
- notes.

This creates a path toward a real-world learning cycle:

```text
Prediction
    ↓
User behavior occurs
    ↓
Actual outcome recorded
    ↓
Feedback dataset grows
    ↓
Retraining
    ↓
New model
    ↓
New predictions
```

This is one of the most important differences between a static ML demo and an evolving ML application.

---

# Multi-Company Architecture

VibeIQ is designed as a **multi-tenant** application.

A tenant is an isolated organization/company using the platform.

For example:

```text
Company A
 ├── users.csv
 ├── predictions
 ├── feedback
 └── models

Company B
 ├── users.csv
 ├── predictions
 ├── feedback
 └── models
```

Each company receives a slug-specific storage area such as:

```text
company_data/<company-slug>/
company_models/<company-slug>/
```

The Django database also associates predictions, datasets, training runs, feedback, and memberships with a company.

The web application uses the logged-in user's company membership to scope the data shown in the interface.

---

# Authentication and API

VibeIQ supports two authentication layers.

## Web authentication

Users can:

- register a company,
- create an account,
- log in,
- log out.

## JWT authentication

The API uses **JWT = JSON Web Token** authentication through Django REST Framework SimpleJWT.

A JWT is a signed token that a client can send with API requests to prove that it has authenticated.

### Obtain an access token

```http
POST /api/auth/token/
```

Request:

```json
{
  "username": "demo_admin",
  "password": "demo1234"
}
```

### Refresh an access token

```http
POST /api/auth/token/refresh/
```

### Get current user/company information

```http
GET /api/me/
```

Header:

```http
Authorization: Bearer <access_token>
```

### Prediction endpoint

```http
POST /api/predict/
```

The API accepts the behavioral feature values and returns prediction outputs.

---

# Dashboard and Application Pages

The major application areas include:

## Dashboard

High-level audience intelligence including:

- audience size,
- churn/risk distribution,
- engagement health,
- segment distribution,
- action distribution,
- LTV information,
- retention priorities.

## Audience

Searchable and filterable user intelligence.

Filters include:

- risk level,
- behavioral segment,
- user ID/search.

## User Detail

Individual user-level intelligence including model outputs and behavioral context.

## Predict

Two prediction paths:

1. upload a user profile file,
2. enter features manually.

The page also presents explainability.

## Model

Model and metric information.

## Train

Company-specific model retraining.

## Test Model

Evaluation against labeled test data.

## Datasets

Dataset Hub containing company-specific datasets and templates.

## Label & Learn

Capture observed outcomes and feedback.

## AI Copilot

Natural-language company-scoped analytics.

---

# Current Demo Dataset

The shipped demo dataset contains:

| Item | Current value |
|---|---:|
| Users | 15,000 |
| Streaming events | 50,000 |
| Model features | 32 |
| Audience prediction rows | 15,000 |
| Behavioral segments | 6 |
| Next-action classes | 7 |
|

### Demo risk distribution

The current shipped predictions contain:

| Risk | Users | Approx. share |
|---|---:|---:|
| LOW | 12,929 | 86.2% |
| MODERATE | 408 | 2.7% |
| HIGH | 749 | 5.0% |
| CRITICAL | 914 | 6.1% |

These are properties of the included **synthetic demonstration dataset**, not real customer statistics.

---

# Current Model Performance

The shipped model report was generated from synthetic data.

The shipped artifacts use a 5,000-row fitting sample from the 15,000-user dataset for the model-training demonstration.

## Churn model

| Metric | Value |
|---|---:|
| Accuracy | 93.6% |
| Precision | 43.56% |
| Recall | 86.27% |
| F1 Score | 57.89% |
| ROC-AUC | 0.9620 |

### What these metrics mean

**Accuracy** — proportion of predictions that are correct overall.

**Precision** — among users predicted as churners, the proportion that are actually churners.

**Recall** — among actual churners, the proportion the model successfully identifies.

**F1 Score** — harmonic mean of precision and recall. It balances both measures.

**ROC-AUC** — Area Under the Receiver Operating Characteristic curve. It measures how well the model ranks positive cases above negative cases across classification thresholds.

For churn/retention systems, recall can be particularly important because missing a true churner can mean losing an opportunity for intervention. However, high recall can increase false positives, which is why precision must also be monitored.

---

## LTV model

| Metric | Value |
|---|---:|
| Mean Absolute Error (MAE) | ₹77.87 |
| Root Mean Squared Error (RMSE) | ₹97.43 |
| R² | 0.9264 |

**MAE = Mean Absolute Error.**

It is the average absolute difference between predicted and observed values.

**RMSE = Root Mean Squared Error.**

It is the square root of the average squared prediction error. Because errors are squared before averaging, large errors receive more weight.

**R² = R-squared, the coefficient of determination.**

It measures how much of the variance in the target is explained by the model relative to a baseline.

---

## Next-action model

| Metric | Value |
|---|---:|
| Accuracy | 76.2% |

The action prediction is a multiclass problem, so the interpretation of accuracy differs from binary churn accuracy: the model must select the correct action among multiple classes.

---

# Project Structure

```text
VibeIQ_Audience_Intelligence_Django_ULTIMATE/
│
├── manage.py
│
├── audience_intelligence/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── predictor/
│   ├── admin.py
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── model_service.py
│   ├── training_service.py
│   ├── views.py
│   ├── migrations/
│   ├── management/
│   │   └── commands/
│   │       └── seed_demo.py
│   ├── static/
│   │   └── app.css
│   └── templates/
│
├── data/
│   ├── users.csv
│   ├── streaming_events.csv
│   ├── audience_predictions.csv
│   ├── training_template.csv
│   └── test_template.csv
│
├── model_artifacts/
│   ├── churn_model.pkl
│   ├── ltv_model.pkl
│   ├── action_model.pkl
│   ├── segment_model.pkl
│   └── model_report.json
│
├── ml/
│   ├── generate_data.py
│   └── train_models_fast.py
│
├── notebooks/
│   └── audience_intelligence_end_to_end.ipynb
│
├── requirements.txt
├── run_mac.sh
├── setup_mac.sh
├── FINAL_MODEL_SUMMARY.md
├── UPDATE_NOTES.md
├── README.md
└── .gitignore
```

---

# Installation on macOS

## 1. Open Terminal

Navigate to the folder where the project is located.

Example:

```bash
cd ~/Desktop
```

## 2. Extract the project

```bash
unzip VibeIQ_Audience_Intelligence_Django_ULTIMATE_FILE_IMPORT.zip
```

## 3. Enter the project

```bash
cd VibeIQ_Audience_Intelligence_Django_ULTIMATE
```

## 4. Create a Python virtual environment

```bash
python3 -m venv .venv
```

A **virtual environment** creates an isolated Python package environment for the project so that its dependencies do not interfere with other projects.

## 5. Activate it

```bash
source .venv/bin/activate
```

## 6. Upgrade pip

```bash
python -m pip install --upgrade pip
```

`pip` is Python's package installer.

## 7. Install dependencies

```bash
pip install -r requirements.txt
```

## 8. Apply Django database migrations

```bash
python manage.py migrate
```

A **migration** is Django's mechanism for creating/updating database tables based on application models.

## 9. Create the demo company and demo data

```bash
python manage.py seed_demo
```

## 10. Run Django checks

```bash
python manage.py check
```

## 11. Start the development server

```bash
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

---

# Demo Login

```text
Username: demo_admin
Password: demo1234
```

These credentials are intended only for the included local demonstration environment.

Do not reuse this password in a real deployment.

---

# Using the Application

A typical workflow is:

```text
1. Login
   ↓
2. Open Dashboard
   ↓
3. Inspect audience risk
   ↓
4. Filter Audience
   ↓
5. Open an individual user
   ↓
6. Review churn + LTV + segment + next action
   ↓
7. Inspect explainability
   ↓
8. Review retention queue
   ↓
9. Ask AI Copilot questions
   ↓
10. Upload labels / feedback
   ↓
11. Retrain model
   ↓
12. Test new model
```

---

# Using Your Own Data

The easiest path is to prepare a CSV containing the 32 required behavioral features.

For training, also include:

```text
churn
```

and optionally:

```text
ltv
next_action
```

For prediction of a single user, you can instead upload a file containing the required features for one profile.

The application validates the required columns and reports missing fields rather than silently inventing values.

---

# Training CSV Requirements

A training dataset must contain all 32 features:

```text
sessions_7d
session_change_7d
listening_minutes_7d
listening_change_7d
skip_rate
save_rate
share_rate
playlist_rate
search_rate
discovery_rate
unique_artists_7d
unique_genres_7d
days_active_14d
avg_session_minutes
subscription_age_days
support_tickets_30d
night_listening_share
completion_rate
avg_daily_sessions
weekend_usage_share
session_gap_hours
artist_concentration
genre_concentration
notification_open_rate
recommendation_click_rate
offline_downloads_30d
playlists_created_30d
social_interactions_30d
device_switch_rate
ad_skip_rate
notification_fatigue
content_diversity
```

Required label:

```text
churn
```

Optional labels:

```text
ltv
next_action
```

Minimum requirement:

- 50 usable rows
- both churn classes must be present

That means the dataset cannot contain only `churn = 0` or only `churn = 1`.

---

# Testing CSV Requirements

Testing requires:

- all 32 features,
- `churn`.

A separate test dataset is preferred over evaluating only on the training data because it provides a more meaningful estimate of generalization.

---

# Supported File Formats

## CSV

**Comma-Separated Values.**

Example:

```csv
user_id,sessions_7d,skip_rate,completion_rate,content_diversity
USER_001,12,0.25,0.71,0.64
```

## XLSX

**Excel Open XML Spreadsheet format.**

Useful when business users maintain profiles in Excel.

## JSON

**JavaScript Object Notation.**

Example:

```json
{
  "user_id": "USER_001",
  "sessions_7d": 12,
  "skip_rate": 0.25,
  "completion_rate": 0.71,
  "content_diversity": 0.64
}
```

For prediction, all required features must be available.

## TXT

Plain text files can contain simple `feature: value` or `feature = value` pairs.

## PDF

**Portable Document Format.**

VibeIQ can extract text from supported text-based PDFs. Scanned image-only PDFs may require OCR and are not guaranteed to produce usable structured features.

## DOCX

**Office Open XML Word document format.**

Text containing feature/value pairs can be parsed.

---

# Technology Stack

## Python

The primary programming language.

## Django

The web application framework used for routing, views, forms, authentication, templates, database models, and application structure.

## Django REST Framework

The framework used to expose API endpoints from Django.

## SimpleJWT

The JWT authentication package used for API access.

## Pandas

Used for tabular data manipulation, CSV processing, feature analysis, and scoring datasets.

## NumPy

Used for numerical computation and array operations.

## Scikit-learn

The main machine learning library used for:

- Random Forest classification,
- Random Forest regression,
- train/test splitting,
- metrics,
- StandardScaler,
- K-Means clustering.

## Joblib

Used to serialize and load trained machine learning artifacts such as `.pkl` files.

## Matplotlib / Seaborn

Used in the ML/notebook workflow for data visualization and analysis.

## SQLite

The default local relational database used by Django in this project.

SQLite is convenient for a portfolio/demo application because it requires no separate database server.

## HTML / CSS / JavaScript

Used to build and enhance the browser-based VibeIQ interface.

---

# Abbreviation Glossary

| Abbreviation | Full form | Meaning in VibeIQ |
|---|---|---|
| AI | Artificial Intelligence | Systems that perform tasks associated with intelligent reasoning or automation |
| API | Application Programming Interface | Programmatic interface for communicating with VibeIQ |
| CSV | Comma-Separated Values | Tabular text data format |
| DOCX | Document Open XML | Microsoft Word document format |
| F1 | F1 Score | Harmonic mean of precision and recall |
| HTTP | Hypertext Transfer Protocol | Protocol used for web/API communication |
| JSON | JavaScript Object Notation | Structured text data format |
| JWT | JSON Web Token | Token format used for API authentication |
| K-Means | K-Means Clustering | Unsupervised clustering algorithm |
| LLM | Large Language Model | AI model designed to understand/generate language; the included Copilot does not require one |
| LTV | Lifetime Value | Estimated value generated by a user over their relationship with a platform |
| MAE | Mean Absolute Error | Average absolute prediction error |
| ML | Machine Learning | Algorithms that learn patterns from data |
| PDF | Portable Document Format | Document file format |
| PR | Precision-Recall | Evaluation relationship between precision and recall across thresholds |
| REST | Representational State Transfer | Common architectural style for web APIs |
| ROC | Receiver Operating Characteristic | Curve showing classifier true-positive vs false-positive tradeoff |
| ROC-AUC | ROC Area Under the Curve | Summary measure of ranking/classification performance |
| RMSE | Root Mean Squared Error | Error metric that penalizes larger errors more strongly |
| R² | R-squared | Coefficient of determination for regression |
| UI | User Interface | What the user sees and interacts with in the application |
| XLS | Excel legacy format | Older Microsoft Excel spreadsheet format |
| XLSX | Excel Open XML Spreadsheet | Modern Microsoft Excel spreadsheet format |
|

---

# Important Model Concepts

## Classification

Classification predicts a category.

Examples in VibeIQ:

```text
Churn → 0 / 1
Next action → PLAY / SAVE / SEARCH / ...
```

## Regression

Regression predicts a numerical value.

Example:

```text
LTV → ₹842.50
```

## Supervised Learning

Supervised learning uses known target labels during training.

VibeIQ uses supervised learning for:

- churn,
- LTV,
- next action.

## Unsupervised Learning

Unsupervised learning finds patterns without a target label.

VibeIQ uses unsupervised learning for:

- behavioral segmentation.

## Ensemble Learning

An ensemble combines multiple models or learners.

Random Forest is an ensemble of decision trees.

## Feature Engineering

Feature engineering means converting raw data into useful variables for machine learning.

The 32 VibeIQ behavioral variables are engineered representations of activity, engagement, content breadth, communication response, and usage patterns.

---

# Important Limitations

## 1. The dataset is synthetic

The included dataset is generated for demonstration.

Therefore:

- the model metrics should not be treated as production performance,
- risk distributions do not represent real customer populations,
- predicted LTV values do not represent actual commercial revenue,
- segment names are demonstration interpretations.

## 2. Correlation is not causation

Explainability outputs show how the trained model responds to feature changes.

They do **not** prove that changing a feature will causally change churn.

For example, if reducing `skip_rate` lowers predicted churn, this does not automatically prove that reducing skip rate will cause the customer to stay.

## 3. Probability is not certainty

A churn probability of 80% does not mean the user definitely churns.

It means the model estimates a high likelihood under the learned data-generating patterns.

## 4. Risk thresholds are business rules

The LOW/MODERATE/HIGH/CRITICAL thresholds are application-level choices.

They should be calibrated against actual business costs and outcomes in a real deployment.

## 5. Local AI Copilot

The included Copilot is a deterministic analytics assistant rather than a full general-purpose conversational LLM.

It intentionally avoids requiring a paid external AI API.

## 6. SQLite is for local/demo use

For production deployment, a more robust database such as PostgreSQL would normally be preferable.

## 7. Production security needs additional hardening

A production deployment should add or review:

- HTTPS,
- secure secret management,
- secure cookie settings,
- CSRF configuration,
- rate limiting,
- stronger password policies,
- centralized logging,
- database backups,
- object storage,
- monitoring,
- model governance,
- privacy controls,
- access auditing.

---

# Future Improvements

Potential extensions include:

## Better feature pipelines

Automatically aggregate event-level data into the 32 features instead of relying on precomputed user-level features.

## Time-series churn modeling

Use sequences of user activity rather than a single feature snapshot.

Possible approaches include:

- gradient boosting,
- recurrent neural networks,
- temporal convolution,
- transformer-based sequence models.

## Better explainability

Potential future methods include:

- SHAP — SHapley Additive exPlanations,
- permutation importance,
- partial dependence,
- accumulated local effects.

## Model monitoring

Track:

- data drift,
- feature drift,
- prediction drift,
- calibration,
- precision/recall over time,
- business intervention outcomes.

## Experimentation

Connect recommendations to controlled experiments to determine whether interventions actually reduce churn.

## Production database

Move from local SQLite to PostgreSQL or another managed relational database.

## Cloud deployment

Possible deployment architecture:

```text
Browser
   ↓
Cloud Load Balancer
   ↓
Django Application
   ↓
PostgreSQL
   ↓
Object Storage
   ↓
Model Registry / Model Store
```

## Advanced Copilot

A future version could connect the Copilot to a controlled LLM while keeping all company-level authorization and data isolation rules in the application layer.

---

# Why This Project Is More Than a Churn Model

A simple churn project usually looks like:

```text
CSV → Train Classifier → Accuracy → Done
```

VibeIQ is intentionally broader:

```text
                         VibeIQ
                            │
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
   Data Layer          ML Layer           Application Layer
        │                   │                   │
        ↓                   ↓                   ↓
   User Events        Churn Model          Django UI
   User Profiles      LTV Model            Dashboard
   Training Data      Action Model         Audience
   Test Data          K-Means              Prediction
                      Segmentation          Copilot
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ↓
                   Explainable Intelligence
                            ↓
                   Retention Prioritization
                            ↓
                     Feedback / Labels
                            ↓
                        Retraining
                            ↓
                       Model Testing
```

The project therefore demonstrates not just model training, but the **ML lifecycle inside a software product**.

---

# Reproducibility

The repository includes:

- source code,
- model artifacts,
- synthetic datasets,
- training templates,
- test templates,
- a notebook,
- dependency definitions,
- migration files,
- demo seed command,
- setup scripts.

The goal is that another developer can clone the repository, install dependencies, migrate the database, seed the demo environment, and run VibeIQ locally without needing private data or an external AI API key.

---

# GitHub Quick Start

After cloning the repository:

```bash
cd VibeIQ_Audience_Intelligence_Django_ULTIMATE
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py seed_demo
python manage.py check
python manage.py runserver
```

Then visit:

```text
http://127.0.0.1:8000/
```

Demo account:

```text
Username: demo_admin
Password: demo1234
```

---

# Final Note

VibeIQ is a demonstration of how **behavioral analytics, machine learning, explainability, business rules, web engineering, authentication, multi-tenancy, feedback, and model retraining** can be combined into a single end-to-end product.

The strongest way to evaluate the project is not only to look at the individual models, but to follow the complete lifecycle:

```text
Data
 ↓
Features
 ↓
Models
 ↓
Predictions
 ↓
Explainability
 ↓
Business Decisions
 ↓
User Feedback
 ↓
Retraining
 ↓
Testing
 ↓
Improved Decisions
```

**VibeIQ — turning audience behavior into actionable intelligence.**
