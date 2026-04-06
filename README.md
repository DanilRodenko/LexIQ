# LexIQ — Legal Contract Analyzer

Fine-tuned DistilBERT model for extractive QA on legal contracts. Given a contract and a question about a specific clause, LexIQ extracts the relevant span directly from the contract text.

## Model

- **Architecture:** DistilBERT (distilbert-base-uncased) fine-tuned for Extractive QA
- **Dataset:** CUAD (Contract Understanding Atticus Dataset) — 510 contracts, 41 clause types, ~20K QA pairs
- **Training:** 3 epochs, AdamW (lr=2e-5), batch size 16
- **Metrics on test set:** Exact Match: 10.75% | F1: 13.37%

## Tech Stack

Python · PyTorch · HuggingFace Transformers · FastAPI · Docker

## Project Structure
```
LexIQ/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_finetuning.ipynb
│   └── 04_evaluation.ipynb
├── models/              # fine-tuned model (not tracked in git)
├── data/                # raw and processed data (not tracked in git)
├── app.py               # FastAPI application
├── Dockerfile
└── requirements_app.txt
```

## Run Locally
```bash
pip install -r requirements_app.txt
uvicorn app:app --reload
```

## Run with Docker
```bash
docker build -t lexiq .
docker run -p 8000:8000 lexiq
```

## API

**POST** `/predict`
```json
{
  "question": "What is the name of this contract?",
  "context": "THIS DISTRIBUTOR AGREEMENT is made by and between..."
}
```

Response:
```json
{
  "answer": "distributor agreement"
}
```

## Author

Danil Rodenko · [GitHub](https://github.com/DanilRodenko) · [LinkedIn](https://linkedin.com/in/danilrodenko)
