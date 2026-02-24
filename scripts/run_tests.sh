#!/usr/bin/env bash
set -euo pipefail
poetry run pytest tests/ -v --asyncio-mode=auto

