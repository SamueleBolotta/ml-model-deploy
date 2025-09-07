FROM python:3.9-slim

WORKDIR /app

# Installa dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia modello e codice applicazione
COPY model.pkl .
COPY app.py .

EXPOSE 8000

# Comando per avviare l'applicazione
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
