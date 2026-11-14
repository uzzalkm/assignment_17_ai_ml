# Heart Disease Prediction API (FastAPI + Docker + Render)

A FastAPI service that serves predictions from a Random Forest classifier trained on the Heart Disease dataset (13 clinical features → presence/absence of heart disease).

## Project structure

```
heart-api/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app: /health, /info, /predict
│   └── schemas.py         # Pydantic input/output models
├── model/
│   └── heart_model.joblib # Trained model (committed so Docker can copy it)
├── data/
│   └── heart.csv          # Training data
├── train.py               # Trains and saves the model
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
```

## 1. Data & model

Dataset: [Kaggle – Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
Columns: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`.

To retrain (e.g. with the Kaggle `heart.csv`):

```bash
pip install -r requirements.txt
python train.py data/heart.csv
```

This saves `model/heart_model.joblib` (a dict with the model, feature list and test accuracy).

## 2. Run locally with Docker

```bash
docker-compose build
docker-compose up
```

Then open Swagger UI: http://localhost:8000/docs

## 3. Endpoints

| Method | Path       | Description                                   |
|--------|------------|-----------------------------------------------|
| GET    | `/health`  | Health check → `{"status": "ok"}`             |
| GET    | `/info`    | Model type, feature list, test accuracy       |
| POST   | `/predict` | Returns `{"heart_disease": true/false, "probability": float}` |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
# {"heart_disease":true,"probability":0.655}
```

## 4. Deploy on Render

1. Push this folder to a GitHub repo:
   ```bash
   git init && git add . && git commit -m "Heart disease FastAPI app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/heart-api.git
   git push -u origin main
   ```
2. On https://dashboard.render.com → **New → Web Service** → connect the repo.
3. Settings:
   - **Language:** Docker
   - **Root Directory / build context:** leave blank (Dockerfile is at the repo root)
   - **Instance type:** Free
4. Click **Create Web Service** and wait for the build. Render sets `$PORT`, and the Dockerfile's `CMD` picks it up automatically.
5. Test the live URL:
   ```bash
   curl https://<your-service>.onrender.com/health
   curl https://<your-service>.onrender.com/info
   curl -X POST https://<your-service>.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
   ```
   Swagger UI: `https://<your-service>.onrender.com/docs`

> Note: free Render instances sleep after inactivity; the first request may take ~30–60 s.

## Notes

- The Docker image uses `python:3.13-slim` with the same pinned library versions used to train the model, so the joblib file loads without version-mismatch warnings. If you retrain with different versions, update `requirements.txt` to match.
