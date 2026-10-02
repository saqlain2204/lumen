"""Local HTTP and websocket server for the Lumen dashboard."""

from __future__ import annotations

import asyncio
import math
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lumen import __version__
from lumen.engine import Engine, Hub
from lumen.store import Store, default_db_path

STATIC = Path(__file__).resolve().parent / "static"


class Claim(BaseModel):
    pid: int
    kind: str = "training"
    model: str | None = None
    name: str = "process"
    command: str = ""


class Ingest(BaseModel):
    run_id: str | None = None
    pid: int | None = None
    type: str = "train"
    metrics: dict[str, float] = Field(default_factory=dict)


def idle_payload(demo: bool) -> dict:
    return {
        "now": time.time(),
        "demo": demo,
        "ready": False,
        "host": {},
        "cpu": {"percent": 0, "per_core": [], "freq_mhz": None},
        "memory": {
            "percent": 0,
            "used": 0,
            "total": 0,
            "swap_percent": 0,
            "swap_used": 0,
            "swap_total": 0,
        },
        "disk": {"read_bps": 0, "write_bps": 0},
        "net": {"recv_bps": 0, "sent_bps": 0},
        "gpus": [],
        "gpu_note": None,
        "runs": [],
        "history": [],
        "metrics": [],
        "ollama": {"online": False, "models": []},
        "run_count": 0,
    }


def _clean_metrics(metrics: dict) -> dict[str, float]:
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if not math.isfinite(number):
            continue
        clean[str(key)[:64]] = number
        if len(clean) >= 40:
            break
    return clean


def create_app(*, demo: bool = False, db_path: str | None = None, start: bool = True) -> FastAPI:
    path = str(db_path or default_db_path())
    store = Store(path)
    hub = Hub()
    engine = Engine(store, hub, demo=demo) if start else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.engine is not None:
            app.state.engine.start()
        yield
        if app.state.engine is not None:
            app.state.engine.stop()
        app.state.store.close()

    app = FastAPI(title="Lumen", version=__version__, lifespan=lifespan)
    app.state.store = store
    app.state.hub = hub
    app.state.engine = engine
    app.state.demo = demo
    app.state.db_path = path

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "demo": demo, "db": path}

    @app.get("/api/live")
    def live():
        payload = hub.latest()
        return payload if payload is not None else idle_payload(demo)

    @app.get("/api/runs")
    def runs():
        return {"runs": store.list_runs()}

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        end = run["ended_at"] or time.time()
        return {
            "run": run,
            "samples": store.samples_between(run["started_at"], end),
            "metrics": store.metrics_for(run_id),
        }

    @app.post("/api/runs/claim")
    def claim(body: Claim):
        if body.pid <= 0:
            raise HTTPException(status_code=400, detail="pid must be a positive integer")
        kind = body.kind if body.kind in {"training", "serving", "inference", "running"} else "running"
        run_id = store.open_run(
            {
                "pid": body.pid,
                "name": body.name or "process",
                "command": body.command,
                "kind": kind,
                "model": body.model,
                "create_time": time.time(),
            },
            overwrite_model=True,
        )
        return {"id": run_id}

    @app.post("/api/ingest")
    def ingest(body: Ingest):
        clean = _clean_metrics(body.metrics)
        if not clean:
            raise HTTPException(status_code=400, detail="metrics need numeric values")
        kind = body.type if body.type in {"train", "infer", "custom"} else "custom"
        if body.run_id:
            if store.get_run(body.run_id) is None:
                raise HTTPException(status_code=404, detail="run not found")
            run_id = body.run_id
        elif body.pid and body.pid > 0:
            mapped = "training" if kind == "train" else "inference"
            run_id = store.open_run(
                {
                    "pid": body.pid,
                    "name": "process",
                    "command": "",
                    "kind": mapped,
                    "model": None,
                    "create_time": time.time(),
                }
            )
        else:
            run_id = _match_open_run(store, kind)
        store.add_metric(time.time(), run_id, kind, clean)
        return {"ok": True, "run_id": run_id}

    @app.websocket("/api/stream")
    async def stream(socket: WebSocket):
        await socket.accept()
        last = None
        try:
            while True:
                payload = hub.latest()
                stamp = payload.get("now") if payload else None
                if payload is not None and stamp != last:
                    await socket.send_json(payload)
                    last = stamp
                await asyncio.sleep(0.25)
        except (WebSocketDisconnect, RuntimeError):
            return

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", _Static(directory=STATIC), name="static")
    return app


def _match_open_run(store: Store, kind: str) -> str | None:
    runs = store.open_runs()
    if not runs:
        return None
    if len(runs) == 1:
        return runs[0]["id"]
    want = "training" if kind == "train" else "inference"
    typed = [
        run
        for run in runs
        if run["kind"] == want or (want == "inference" and run["kind"] == "serving")
    ]
    if len(typed) == 1:
        return typed[0]["id"]
    return runs[0]["id"]


class _Static(StaticFiles):
    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response
