FROM python:3.9.18-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 系統層相依（你原本的保留）
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Render 會自動用這個 port
EXPOSE 10000

# 🔴 關鍵修改在這一行
CMD ["gunicorn", "-b", "0.0.0.0:10000", "--timeout", "3000", "--workers", "1", "app:app"]

