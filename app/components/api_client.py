import os
import requests
from typing import Any, Dict

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://banking-risk-app-mukosi.onrender.com"
).rstrip("/")


def api_get(endpoint: str, timeout: int = 60) -> Any:
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def api_post(endpoint: str, payload: Dict[str, Any], timeout: int = 60) -> Any:
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}