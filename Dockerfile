FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and pre-trained artifacts only
COPY SRC/ ./SRC/
COPY Models/ ./Models/
COPY preprocessor.joblib .

EXPOSE 8000

CMD ["uvicorn", "SRC.predict:app", "--host", "0.0.0.0", "--port", "8000"]