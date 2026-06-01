# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python scripts
COPY extract_features.py .
COPY train_lora.py .
COPY run_inference.py .

# Run scripts in order
CMD ["sh", "-c", "python extract_features.py && python train_lora.py && python run_inference.py"]
