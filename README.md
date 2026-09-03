
# Hybrid GNN Solvation Free Energy Prediction API

## Project Structure

```
Hybrid_GNN_API/
│
├── app.py
├── predictor.py
├── model.py
├── feature_generator.py
├── requirements.txt
├── Dockerfile
├── README.md
│
├── models/
│   ├── best_model.pt
│   ├── pipeline.pkl
│   ├── model_config.pkl
│   ├── feature_order.pkl
│   ├── scaler.pkl
│   └── metrics.pkl
```

---

## Create Environment

```bash
conda create -n hybrid_api python=3.12
conda activate hybrid_api
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run API

```bash
uvicorn app:app --reload
```

API:

http://127.0.0.1:8000/docs

---

## Single Prediction

POST /predict

Example JSON

{
    "solute_smiles":"CCO",
    "solvent_smiles":"O"
}

---

## Docker

Build

docker build -t hybrid-gnn-api .

Run

docker run -p 8000:8000 hybrid-gnn-api

---

## Google Cloud Run

gcloud builds submit --tag gcr.io/PROJECT_ID/hybrid-gnn-api

gcloud run deploy hybrid-gnn-api     --image gcr.io/PROJECT_ID/hybrid-gnn-api     --platform managed     --allow-unauthenticated

