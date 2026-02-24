#!/usr/bin/env bash
set -euo pipefail
poetry run uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
