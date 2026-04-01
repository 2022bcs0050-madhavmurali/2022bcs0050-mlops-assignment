# Use Python 3.9 slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements_inference.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements_inference.txt

# Create the directory structure exactly as expected
COPY models/ ./models/
COPY src/ ./src/

# Expose port 8000
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "src.inference.service:app", "--host", "0.0.0.0", "--port", "8000"]
