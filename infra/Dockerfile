# Dockerfile for YOLOv8 inference with rasterio + S3
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set noninteractive
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git build-essential \
    gdal-bin libgdal-dev libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# GDAL env for rasterio
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Create app dir
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt

# Copy script
COPY scripts/infer_pv_s3.py /app/infer_pv_s3.py
RUN chmod +x /app/infer_pv_s3.py

ENTRYPOINT ["python3", "/app/infer_pv_s3.py"]
