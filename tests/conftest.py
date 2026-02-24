from __future__ import annotations

import pytest
import httpx

from src.api.app import create_app


@pytest.fixture()
async def client() -> httpx.AsyncClient:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client
