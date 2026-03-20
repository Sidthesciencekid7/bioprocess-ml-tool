# 🧬 Bioprocess ML Report Generator

A production-ready Streamlit app for bioprocess run analysis — CHO, microbial, fermentation, or any time-series bioprocess data.

## Features

- **Dual ML models** — Random Forest + XGBoost with cross-validated metrics
- **SHAP explanations** — Per-feature impact beyond simple importance scores
- **Smart process flags** — Configurable thresholds for NH3, glucose, VCD, L/G ratio, spike detection
- **Correlation heatmap** — Identify redundant or co-varying features
- **Multi-run overlay plots** — Compare batches side-by-side
- **PDF + Excel reports** — With timestamps in filenames
- **Works with any dataset** — Auto-detects target, time, and group columns

## Deploy to Streamlit Community Cloud (free, always online)

1. **Push to GitHub:**
   ```bash
   git init
   git add bioprocess_app.py requirements.txt
   git commit -m "Initial commit"
   gh repo create bioprocess-reporter --public --push
   ```

2. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **New app** → select your repo → set main file to `bioprocess_app.py`
   - Click **Deploy** — your app is live in ~2 minutes

3. **Your app URL:** `https://<your-username>-bioprocess-reporter.streamlit.app`

## Run locally

```bash
pip install -r requirements.txt
streamlit run bioprocess_app.py
```

## Input Format

Works with:
- **CDPpy Excel format** (auto-detects header in row 0)
- Any `.csv` or `.xlsx` with numeric columns
- CDPpy sheet: `Measured Data`

## Sidebar Controls

All thresholds are adjustable live without retraining:
- NH3 max threshold
- Glucose min threshold  
- Min peak VCD
- Lactate/Glucose ratio alert
- Number of trees (model complexity)
- SHAP toggle (disable for speed)
