# Menggunakan base image Python 3.10 agar kompatibel dengan pip terbaru
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependensi sistem yang dibutuhkan
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Salin semua file dari folder MLProject ke dalam container
COPY . .

# Install library yang diperlukan
RUN pip install mlflow==2.19.0 scikit-learn pandas numpy matplotlib seaborn

# Command untuk menjalankan model
ENTRYPOINT ["python", "modelling.py"]
