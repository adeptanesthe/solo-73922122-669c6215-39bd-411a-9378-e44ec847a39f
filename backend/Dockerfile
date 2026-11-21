FROM python:3.11-slim

WORKDIR /app

# 先装依赖，利用镜像层缓存
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 后端代码 + 前端静态文件
COPY backend/app ./app
COPY frontend ./frontend

ENV STATIC_DIR=/app/frontend \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
