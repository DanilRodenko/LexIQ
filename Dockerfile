FROM python:3.11-slim

WORKDIR /app

COPY requirements_app.txt .
RUN pip install -r requirements_app.txt

COPY app.py .
COPY models/ ./models/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]