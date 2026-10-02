"""Local runtime probes that stay on the machine."""

from __future__ import annotations

import json
import urllib.request


def parse_ollama_ps(data: dict) -> list[dict]:
    models = []
    for item in data.get("models") or []:
        if not isinstance(item, dict):
            continue
        models.append(
            {
                "name": item.get("name") or item.get("model") or "model",
                "size": item.get("size"),
                "vram": item.get("size_vram"),
            }
        )
    return models


def poll_ollama(host: str = "http://127.0.0.1:11434", timeout: float = 0.25) -> dict:
    url = host.rstrip("/") + "/api/ps"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
        if not isinstance(payload, dict):
            payload = {}
        return {"online": True, "models": parse_ollama_ps(payload)}
    except Exception:
        return {"online": False, "models": []}
