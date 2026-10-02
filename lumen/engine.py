"""Sampling loop that turns hardware snapshots and process scans into a live payload."""

from __future__ import annotations

import os
import threading
import time
import traceback
from collections import deque

import psutil

from lumen.collector import Collector
from lumen.demo import DemoSource
from lumen.detector import _norm, scan_processes
from lumen.integrations import poll_ollama
from lumen.store import Store


class Hub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.payload = None

    def publish(self, payload: dict) -> None:
        with self._lock:
            self.payload = payload

    def latest(self):
        with self._lock:
            return self.payload


def light_gpu(gpu: dict) -> dict:
    total = gpu.get("mem_total") or 0
    used = gpu.get("mem_used") or 0
    vram = (100.0 * used / total) if total else None
    return {
        "util": gpu.get("util"),
        "vram": vram,
        "power": gpu.get("power_w"),
        "temp": gpu.get("temp"),
        "power_limit": gpu.get("power_limit_w"),
    }


def summarize_gpus(gpus: list[dict]) -> dict:
    if not gpus:
        return {
            "gpu_util": None,
            "vram_percent": None,
            "vram_used": None,
            "vram_total": None,
            "power_w": None,
            "temp_c": None,
            "name": None,
        }
    used = sum(gpu.get("mem_used") or 0 for gpu in gpus)
    total = sum(gpu.get("mem_total") or 0 for gpu in gpus)
    powers = [gpu.get("power_w") for gpu in gpus if gpu.get("power_w") is not None]
    temps = [gpu.get("temp") for gpu in gpus if gpu.get("temp") is not None]
    utils = [gpu.get("util") for gpu in gpus if gpu.get("util") is not None]
    return {
        "gpu_util": max(utils) if utils else None,
        "vram_percent": (100.0 * used / total) if total else None,
        "vram_used": used,
        "vram_total": total,
        "power_w": sum(powers) if powers else None,
        "temp_c": max(temps) if temps else None,
        "name": " · ".join(gpu.get("name") for gpu in gpus if gpu.get("name")) or None,
    }


class Engine:
    def __init__(self, store: Store, hub: Hub, demo: bool = False):
        self.store = store
        self.hub = hub
        self.demo = demo
        self.history: deque = deque(maxlen=180)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.misses: dict[int, int] = {}
        self.ollama = {"online": False, "models": []}
        self._next_ollama = 0.0
        self.collector = None if demo else Collector()
        self.demo_source = DemoSource() if demo else None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        if self.collector is not None:
            self.collector.prime()
        self.store.prune()
        self._reap()
        if self.demo_source is not None:
            begin = time.time() - 180
            for offset in range(180):
                self.tick(now=begin + offset)
        else:
            self.tick()
        self.thread = threading.Thread(target=self._loop, name="lumen-collector", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)

    def _loop(self) -> None:
        while not self.stop_event.wait(1.0):
            try:
                self.tick()
            except Exception:
                traceback.print_exc()

    def _reap(self) -> None:
        for run in self.store.open_runs():
            pid = run.get("pid")
            if not pid or not psutil.pid_exists(int(pid)):
                self.store.close_run(run["id"])

    def _poll_ollama(self, now: float) -> dict:
        if now < self._next_ollama:
            return self.ollama
        self.ollama = poll_ollama()
        self._next_ollama = now + (2.0 if self.ollama["online"] else 5.0)
        return self.ollama

    def tick(self, now: float | None = None) -> None:
        ts = time.time() if now is None else now
        if self.demo_source is not None:
            snap, procs, metric = self.demo_source.tick(ts)
            ollama = {"online": False, "models": []}
        else:
            assert self.collector is not None
            snap = self.collector.snapshot()
            procs = scan_processes(exclude_pids={os.getpid()})
            metric = None
            ollama = self._poll_ollama(ts)
            procs = [self._decorate(proc, ollama) for proc in procs]
        self._sync(procs, ts)
        if metric and procs:
            run_id = self.store.open_run(procs[0], ts=ts)
            self.store.add_metric(ts, run_id, "train", metric)
        summary = summarize_gpus(snap.get("gpus") or [])
        memory = snap.get("memory") or {}
        disk = snap.get("disk") or {}
        light = [light_gpu(gpu) for gpu in snap.get("gpus") or []]
        self.history.append({"ts": ts, "cpu": (snap.get("cpu") or {}).get("percent"), "ram": memory.get("percent"), "gpus": light})
        self.store.add_sample(
            {
                "ts": ts,
                "cpu": (snap.get("cpu") or {}).get("percent"),
                "ram": memory.get("percent"),
                "ram_used": memory.get("used"),
                "ram_total": memory.get("total"),
                "gpu_util": summary["gpu_util"],
                "vram_percent": summary["vram_percent"],
                "vram_used": summary["vram_used"],
                "vram_total": summary["vram_total"],
                "power_w": summary["power_w"],
                "temp_c": summary["temp_c"],
                "disk_read": disk.get("read_bps"),
                "disk_write": disk.get("write_bps"),
                "gpus": light,
                "gpu_name": summary["name"],
            }
        )
        self.hub.publish(self._payload(ts, snap, ollama))

    def _decorate(self, proc: dict, ollama: dict) -> dict:
        if _norm(proc.get("name") or "") == "ollama" and not proc.get("model"):
            names = [item["name"] for item in ollama.get("models") or [] if item.get("name")]
            if names:
                proc = dict(proc)
                proc["model"] = ", ".join(names)
        return proc

    def _sync(self, procs: list[dict], ts: float) -> None:
        current = {proc["pid"]: proc for proc in procs}
        open_runs = {run["pid"]: run for run in self.store.open_runs()}
        for pid, proc in current.items():
            if pid in open_runs:
                overwrite = _norm(proc.get("name") or "") == "ollama"
                self.store.update_run(open_runs[pid]["id"], proc, overwrite_model=overwrite)
            else:
                self.store.open_run(proc, ts=ts)
        if self.demo:
            return
        for run in self.store.open_runs():
            pid = run["pid"]
            if pid in current:
                self.misses.pop(pid, None)
                continue
            if pid and psutil.pid_exists(int(pid)):
                self.misses.pop(pid, None)
                continue
            self.misses[pid] = self.misses.get(pid, 0) + 1
            if self.misses[pid] >= 2:
                self.store.close_run(run["id"])
                self.misses.pop(pid, None)

    def _payload(self, ts: float, snap: dict, ollama: dict) -> dict:
        runs = []
        for run in self.store.open_runs():
            tail = self.store.metrics_for(run["id"], limit=80)
            losses = [item["data"]["loss"] for item in tail if isinstance(item["data"].get("loss"), (int, float))]
            run["loss_tail"] = losses
            run["last_metric"] = tail[-1]["data"] if tail else None
            runs.append(run)
        return {
            "now": ts,
            "demo": self.demo,
            "ready": True,
            "host": snap.get("host") or {},
            "cpu": snap.get("cpu") or {},
            "memory": snap.get("memory") or {},
            "disk": snap.get("disk") or {},
            "net": snap.get("net") or {},
            "gpus": snap.get("gpus") or [],
            "gpu_note": snap.get("gpu_note"),
            "runs": runs,
            "history": list(self.history),
            "metrics": self.store.recent_metrics(8),
            "ollama": ollama,
            "run_count": self.store.count_runs(),
        }
