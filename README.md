# US Flight Operational Risk Analysis Setup (Web App)

This repository contains the Machine Learning Application predicting flight delays and modeling severe operational risks.

## Architecture

- **Frontend**: React (`/client`)
- **Backend**: FastAPI (`/api`)
- **Machine Learning Models**: Scikit-Learn (Hosted remotely on Hugging Face)

## Prerequisites

- Node.js
- Python 3.8+

## API Setup (Backend)

The backend dynamically downloads the required multi-stage Machine Learning models from a remote Hugging Face repository upon first startup to keep the GitHub repository incredibly lightweight.

1. Navigate to the API folder

```bash
cd api
```

2. Install Python dependencies

```bash
pip install -r requirements.txt
```

3. Start the FastAPI Server (This will automatically download the models on first boot)

```bash
uvicorn main:app --reload
```

## Client Setup (Frontend)

1. Open a new terminal and navigate to the client folder

```bash
cd client
```

2. Install npm dependencies

```bash
npm install
```

3. Start the development server

```bash
npm run dev
```

_Note: The raw data analysis notebooks and scripts used to train the original models are preserved in the `archive` commit history for reference._
