.PHONY: run test lint install clean docker-build docker-run

run:
	poetry run uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000

test:
	poetry run pytest tests/ -v --asyncio-mode=auto

lint:
	poetry run python -m py_compile $$(find src -name "*.py")

install:
	poetry install

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

docker-build:
	docker build -t ai-quantum-platform:latest .

docker-run:
	docker-compose up -d
