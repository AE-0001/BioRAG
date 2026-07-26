FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data/demo ./data/demo
COPY app.py ./
RUN pip install --no-cache-dir ".[ui]"

EXPOSE 8000
CMD ["uvicorn", "biorag.api:app", "--host", "0.0.0.0", "--port", "8000"]
