# FraudShield-AI Deployment Guide

This guide helps you deploy FraudShield-AI to Streamlit Cloud with MongoDB Atlas integration.

---

## 🚀 Quick Streamlit Cloud Deployment

### Step 1: Prepare Your GitHub Repository
✅ Your code is already on GitHub at: `https://github.com/Nishant-sonar/FraudShield-AI`

### Step 2: Deploy on Streamlit Cloud

1. Go to: **https://share.streamlit.io/**
2. Sign in with GitHub (if not already logged in)
3. Click **"New app"** button
4. Fill in the form:
   - **Repository**: `Nishant-sonar/FraudShield-AI`
   - **Branch**: `main`
   - **Main file path**: `app_professional_dashboard.py`
5. Click **"Deploy"**

Your app will be live at:
```
https://share.streamlit.io/Nishant-sonar/FraudShield-AI
```

---

## 🔐 Configure MongoDB Atlas Connection (IMPORTANT!)

### Without This Step: App Will Work Locally But Fail on Streamlit Cloud

### Step 1: Add Secrets to Streamlit Cloud

On your app's deployed page:
1. Click **⋮** (three dots) → **Settings**
2. Click **"Secrets"** tab
3. Paste your MongoDB connection string:

```toml
MONGO_URI = "mongodb+srv://nishantmongodb:Java@Python#2005@fraud-detection-cluster.rvmfsyy.mongodb.net/?appName=Fraud-detection-cluster"
```

4. Click **"Save"**

---

## 🏠 Local Development Setup

### Using Local MongoDB
```bash
# Install MongoDB Community Edition
# Start MongoDB
mongod

# Run the app
streamlit run app_professional_dashboard.py
```

### Using MongoDB Atlas (Cloud)

Create `.streamlit/secrets.toml`:
```toml
MONGO_URI = "mongodb+srv://nishantmongodb:Java@Python#2005@fraud-detection-cluster.rvmfsyy.mongodb.net/?appName=Fraud-detection-cluster"
```

Then:
```bash
streamlit run app_professional_dashboard.py
```

**Note:** `.streamlit/secrets.toml` is in `.gitignore` — it won't be committed to GitHub.

---

## 📋 Troubleshooting

### "MongoDB connection failed" on Streamlit Cloud

→ Check if secrets are added in Streamlit Cloud dashboard

### "Cannot read model files"

→ Ensure `artifacts/model.pkl` is committed to GitHub

### "Missing dependencies"

→ All packages are in `requirements.txt` — Streamlit Cloud installs automatically

---

## 🎯 Verification

After deployment, test with these inputs:
- **Transaction Amount**: 5000
- **Sender Bank ID**: 12345
- **Receiver Bank ID**: 67890
- **Currency**: USD

Expected: Shows fraud prediction with SHAP explanation

---

## 📚 Additional Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud/get-started)
- [MongoDB Atlas Free Tier](https://www.mongodb.com/cloud/atlas/register)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)

---

**Deployment Status**: ✅ Ready for Production
