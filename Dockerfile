# Use Python 3.11 slim image 
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code 
COPY app/ ./app/
COPY models/ ./models/


# Expose port 
EXPOSE 2020

# Run the application 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "2020", "--reload"]