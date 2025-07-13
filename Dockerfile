# ─── Dockerfile ────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your full app code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Start the dashboard by default
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
