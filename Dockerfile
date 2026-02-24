FROM python:3.11-slim

WORKDIR /app
RUN pip install poetry==1.8.0

COPY pyproject.toml ./
RUN poetry config virtualenvs.create false && poetry install --no-interaction

COPY src/ ./src/

RUN mkdir -p /workspace/projects
ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]

