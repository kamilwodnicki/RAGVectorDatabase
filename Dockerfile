FROM python:3.10-slim

# CUDA_VARIANT wybiera plik requirements: cu118 | cu126 | cpu
# Ustaw w .env: CUDA_VARIANT=cu118  (domyślnie cu118)
ARG CUDA_VARIANT=cu118

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/ requirements/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements/${CUDA_VARIANT}.txt

COPY . .

RUN mkdir -p DOKUMENTY

CMD ["python", "manage.py", "serve"]