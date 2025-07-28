# Python runtime
FROM python:3.10-slim

# We set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .
COPY setup.py .
COPY README.md .

# We install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# we the copy the entire project
COPY . .

# Critical: Add src to Python path so 'from defaultMlProj import ...' works
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Run FastAPI with uvicorn
# Use shell form to expand $PORT
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT

