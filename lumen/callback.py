"""Optional hooks that push training and inference numbers into a running Lumen."""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.request

try:
    from transformers import TrainerCallback as _TrainerCallback
except Exception:  # transformers is optional; the class still imports
    class _TrainerCallback:  # type: ignore[no-redef]
        pass


def _endpoint(url: str | None) -> str:
    raw = url or os.environ.get("LUMEN_URL") or "http://127.0.0.1:8787"
    return raw.rstrip("/")


def _post(url: str, payload: dict, timeout: float = 0.4) -> bool:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def _finite(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif hasattr(value, "item"):
        try:
            number = float(value.item())
        except (TypeError, ValueError):
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def log_metrics(
    metrics: dict,
    *,
    type: str = "train",
    pid: int | None = None,
    run_id: str | None = None,
    url: str | None = None,
) -> bool:
    """Send numeric metrics to the local Lumen server.

    Returns False when Lumen is not running. Never raises.
    """
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        number = _finite(value)
        if number is None:
            continue
        clean[str(key)[:64]] = number
        if len(clean) >= 40:
            break
    if not clean:
        return False
    body: dict = {"type": type, "metrics": clean}
    if pid is not None:
        body["pid"] = pid
    if run_id is not None:
        body["run_id"] = run_id
    return _post(_endpoint(url) + "/api/ingest", body)


def log_inference(
    *,
    model: str | None = None,
    tokens_per_sec: float | None = None,
    ttft_ms: float | None = None,
    prompt_tokens: float | None = None,
    completion_tokens: float | None = None,
    url: str | None = None,
    **extra: float,
) -> bool:
    """Record one inference measurement and, when a model name is given, open a session."""
    base = _endpoint(url)
    if model:
        _post(
            base + "/api/runs/claim",
            {
                "pid": os.getpid(),
                "kind": "inference",
                "model": model,
                "name": "inference",
                "command": " ".join(sys.argv),
            },
        )
    metrics: dict = {}
    if tokens_per_sec is not None:
        metrics["tokens_per_sec"] = tokens_per_sec
    if ttft_ms is not None:
        metrics["ttft_ms"] = ttft_ms
    if prompt_tokens is not None:
        metrics["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        metrics["completion_tokens"] = completion_tokens
    metrics.update(extra)
    return log_metrics(metrics, type="infer", pid=os.getpid(), url=base)


class LumenCallback(_TrainerCallback):
    """Hugging Face Trainer callback.

    Training continues unchanged when Lumen is not running.
    """

    def __init__(self, url: str | None = None, model: str | None = None):
        self.url = _endpoint(url)
        self.model = model

    def _model_name(self, model) -> str | None:
        if self.model:
            return self.model
        if model is None:
            return None
        named = getattr(model, "name_or_path", None)
        if named:
            return str(named)
        config = getattr(model, "config", None)
        named = getattr(config, "name_or_path", None) if config is not None else None
        if named:
            return str(named)
        return model.__class__.__name__

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        _post(
            self.url + "/api/runs/claim",
            {
                "pid": os.getpid(),
                "kind": "training",
                "model": self._model_name(model),
                "name": "Hugging Face Trainer",
                "command": " ".join(sys.argv),
            },
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        metrics: dict[str, float] = {}
        for key, value in logs.items():
            number = _finite(value)
            if number is not None:
                metrics[str(key)] = number
        step = getattr(state, "global_step", None)
        if step is not None and "step" not in metrics:
            metrics["step"] = float(step)
        if metrics:
            log_metrics(metrics, type="train", pid=os.getpid(), url=self.url)
