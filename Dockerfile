FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system deps + ODBC + wget
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    gcc \
    g++ \
    unixodbc \
    unixodbc-dev \
    apt-transport-https \
    wget \
    && curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p /app/models

# 🔥 Download models from Google Drive
RUN wget -O /app/models/credit_model_best.pkl "https://drive.google.com/uc?export=download&id=13xKr7FBBEgN2LqfAkvOWmnoIRmS4Lz0p"
RUN wget -O /app/models/fraud_model_best.pkl "https://drive.google.com/uc?export=download&id=1maIHAtaLHIV7xGHg-_SjL2w2Thxg0VI6"

# Copy project
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]