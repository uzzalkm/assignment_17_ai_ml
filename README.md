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

