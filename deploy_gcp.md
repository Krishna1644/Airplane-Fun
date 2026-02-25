# Deploying Flight Risk Evaluator to Google Cloud Run

This guide will walk you through containerizing and deploying your application to Google Cloud Run, a fully managed compute environment that scales your containers automatically.

## Prerequisites

1. [Google Cloud SDK (gcloud CLI)](https://cloud.google.com/sdk/docs/install) installed and authenticated.
2. A Google Cloud Project created with billing enabled.
3. Docker installed and running locally.

## Initial Setup

Open your terminal and authenticate with Google Cloud, then set your target project ID:

```bash
gcloud auth login
gcloud config set project YOUR_INITIALIZED_PROJECT_ID
```

_Replace `YOUR_INITIALIZED_PROJECT_ID` with your actual GCP Project ID._

Enable the required API services for Cloud Run and Artifact Registry:

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

---

## Default Region Setup

Before starting, set your default compute region (e.g., `us-central1`):

```bash
gcloud config set run/region us-central1
```

---

## 🚀 Step 1: Deploy the Backend (FastAPI)

1. Navigate to the `api` directory:

```bash
cd api
```

2. Build and deploy to Cloud Run using Google Cloud Build in one command:

```bash
gcloud run deploy flight-risk-backend `
  --source . `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 1
```

_Note: We request 2Gi of memory here because loading the large `joblib` models from Hugging Face requires sufficient RAM._

3. Success! The terminal will output a **Service URL** (e.g., `https://flight-risk-backend-xxxxx-uc.a.run.app`).
   **Copy this URL**, you will need it for the frontend.

---

## 🚀 Step 2: Deploy the Frontend (React Vite)

1. Navigate to the `client` directory:

```bash
cd ../client
```

2. We need to build the React app and inject the backend URL we just created. Run the following deployment command, replacing `[YOUR_BACKEND_URL]` with the URL you copied in Step 1.

```bash
gcloud build submits --tag gcr.io/YOUR_INITIALIZED_PROJECT_ID/flight-risk-frontend . --build-arg=VITE_API_BASE_URL=[YOUR_BACKEND_URL]
```

_(Make sure you replace BOTH placeholders above, and ensure there is NO trailing slash `/` on your backend url)_

3. Once built, deploy the image to Cloud Run:

```bash
gcloud run deploy flight-risk-frontend `
  --image gcr.io/YOUR_INITIALIZED_PROJECT_ID/flight-risk-frontend `
  --allow-unauthenticated `
  --port 80
```

4. You will receive a second **Service URL** for the frontend. Click it to view your live, production-ready Flight Risk Evaluator on the public internet! 🚀
