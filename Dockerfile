FROM python:3.11-slim

# Install system dependencies (git, build-essential for LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
        libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Entry point
CMD ["streamlit", "run", "ui.py", "--server.port", "8501", "--server.headless", "true"]
