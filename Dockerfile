FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system deps + ODBC
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    gcc \
    g++ \
    unixodbc \
    unixodbc-dev \
    apt-transport-https \
    && curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create models folder
RUN mkdir -p /app/models

# 🔥 Use gdown instead of wget
RUN python -m gdown https://drive.google.com/uc?id=1g790wSduI04H0r97lyDmbJpqUUtBYZhV -O /app/models/credit_model_best.pkl
RUN python -m gdown https://drive.google.com/uc?id=1jv4Q7uWob7VU8i1mKLizB6tjK3tly_Be -O /app/models/fraud_model_best.pkl

# Copy project
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]