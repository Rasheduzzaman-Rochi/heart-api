# Heart Disease Prediction API

FastAPI service that serves predictions from a Random Forest classifier trained on the Heart Disease dataset. Dockerized and deployed on Render.

**Live URL:** https://heart-api-n640.onrender.com (Swagger: `/docs`)

## Project structure

```
.
├── app/
│   ├── main.py          # FastAPI app and endpoints
│   └── schemas.py       # Pydantic input/output models
├── model/
│   └── heart_model.joblib
├── data/
│   └── heart.csv        # dataset
├── train.py             # trains and saves the model
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Data and model

- Dataset: [Heart Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). 13 features, binary `target` (1 = disease, 0 = no disease).
- Model: `RandomForestClassifier` (200 trees), 80/20 stratified split.
- Saved with joblib to `model/heart_model.joblib` along with the feature list and test accuracy.

Retrain:

```bash
pip install -r requirements.txt
python train.py
```

To use the Kaggle CSV, put it at `data/heart.csv` and run `train.py` again. The columns are the same.

## Endpoints

| Method | Path       | Description                           |
|--------|------------|---------------------------------------|
| GET    | `/health`  | Health check                          |
| GET    | `/info`    | Model type, feature list, accuracy    |
| POST   | `/predict` | Returns `heart_disease: true/false`   |

Input fields: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`

## Run locally with Docker

```bash
docker-compose build
docker-compose up
```

Open http://localhost:8000/docs

## Test

```bash
curl http://localhost:8000/health
curl http://localhost:8000/info

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}'
```

Response:

```json
{"heart_disease": false, "probability": 0.1}
```

For the live service, replace `http://localhost:8000` with the Render URL.

## Deploy on Render

1. Push this repo to GitHub.
2. Render → New → Web Service → connect the repo.
3. Language: **Docker**. Root directory: leave empty (Dockerfile is at the root).
4. Instance type: Free. Create the service.
5. Render sets `PORT` automatically and the container listens on it.
6. When the deploy is live, test with the curl commands above.

Note: the free plan sleeps after inactivity, so the first request can take around 30-60 seconds.
