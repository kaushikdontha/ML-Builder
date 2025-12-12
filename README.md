# ML Builder - No-Code Machine Learning Platform

ML Builder is a web application that allows users to build, train, and visualize machine learning models without writing code.

## features

-   **Drag & Drop Data**: Upload CSV/Excel datasets.
-   **Intelligent Visualizations**:
    -   Target Distribution (Class Balance)
    -   Correlation Heatmap
    -   Interactive Scatter Plots
-   **No-Code Pipeline**:
    -   Add preprocessing (Scaling, Null Imputation).
    -   Configure Train/Test Split.
    -   Select Models (Logistic Regression, Decision Tree).
-   **Explainable AI**:
    -   Confusion Matrix & Feature Importance charts.
    -   Natural Language Result Interpretation.
-   **Export**: Download trained models as `.pkl` files.

## Tech Stack

-   **Frontend**: Next.js (React), TypeScript.
-   **Backend**: FastAPI (Python), Pandas, Scikit-learn.
-   **ML**: Scikit-Learn.

## Getting Started

### 1. Backend Setup
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
python main.py
```
Backend runs on `http://localhost:8000`.

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Frontend runs on `http://localhost:3000`.

## Deployment
See `deployment_guide.md` for instructions on deploying to **Render** and **Vercel**.
