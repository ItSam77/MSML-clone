FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

RUN mkdir -p /app/model

# Expose port
EXPOSE 5000

# Run MLflow server
CMD ["mlflow", "models", "serve", "-m", "./model", "-h", "0.0.0.0", "-p", "5000", "--no-conda"]