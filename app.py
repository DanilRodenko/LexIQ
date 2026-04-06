import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering

app = FastAPI(title="LexIQ")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

tokenizer = DistilBertTokenizerFast.from_pretrained('models')
model = DistilBertForQuestionAnswering.from_pretrained('models')
model = model.to(device)
model.eval()

class PredictRequest(BaseModel):
    question: str
    context: str

@app.get("/")
def root():
    return {"message": "LexIQ API is running"}

@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    encoding = tokenizer(request.question, request.context, max_length=512, truncation=True, return_offsets_mapping=True)

    input_ids = torch.tensor(encoding['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoding['attention_mask']).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_idx = torch.argmax(outputs.start_logits, dim=1).item()
        end_idx = torch.argmax(outputs.end_logits, dim=1).item()
        answer_tokens = encoding['input_ids'][start_idx:end_idx+1]
        answer = tokenizer.decode(answer_tokens)
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return {"answer": answer}