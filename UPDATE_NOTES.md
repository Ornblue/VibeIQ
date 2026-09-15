# VibeIQ Update

## Audience filtering
The Audience page supports search, risk filters (LOW/MODERATE/HIGH/CRITICAL), and segment filters. It shows the filtered population rather than only a hard-coded critical list.

## File-to-prediction extraction
The Predict page now accepts a user profile file instead of requiring manual entry. Supported formats:
- CSV
- XLSX/XLS
- JSON
- TXT
- PDF
- DOCX

CSV/XLSX/JSON should contain one user row with the 32 model features. TXT/PDF/DOCX can use `feature: value` or `feature = value` lines. The extractor also recognizes common aliases such as `sessions`, `skip`, `save`, `completion`, `recommendation_clicks`, etc.

The first row is analyzed for tabular files. Missing required features produce a clear error instead of a partial prediction.

## Explainability
The Predict page now includes six views:
1. Signed churn-driver bar chart
2. Audience-percentile radar profile
3. Current vs median-feature counterfactual sensitivity
4. Risk-driver vs protective-impact balance
5. Top risk contributors
6. Protective signals

These are model explanations/counterfactual comparisons, not causal claims.
