# AeroCast — M1 Data Ingestion Service Dockerfile
FROM python:3.12-slim

# Install system dependencies for C extensions, GDAL, GEOS, and PROJ (required for geopandas and rasterio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Ensure Python doesn't buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data assets
COPY . .

# Create cache and data directories with appropriate permissions
RUN mkdir -p .cache data/boundaries data/rasters

# Default entrypoint runs the ingestion daemon
ENTRYPOINT ["python", "main.py"]
CMD []
