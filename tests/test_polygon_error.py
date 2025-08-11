from __future__ import annotations

import httpx
import pytest
import types

from trading_agent_v2.adapters.polygon_client import PolygonClient, PolygonError


def test_polygon_error_on_4xx(monkeypatch):
    client = PolygonClient(api_key="bGfLOj1ATBptvynqwAEcW2QXMVne6HEO", base_url="https://api.polygon.io")

    class Resp:
        def __init__(self, status_code=404):
            self.status_code = status_code
            self.headers = {}
            self.text = "not found"

        def json(self):
            return {}

        def raise_for_status(self):
            raise httpx.HTTPStatusError("err", request=None, response=httpx.Response(self.status_code))

    def fake_request(method, url, params=None):
        return Resp(404)

    monkeypatch.setattr(client, "_get_client", lambda: types.SimpleNamespace(request=fake_request))

    with pytest.raises(PolygonError):
        client.get_market_status() 