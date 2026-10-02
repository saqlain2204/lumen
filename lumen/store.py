"""SQLite storage for machine samples, runs, and trainer metrics."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from pathlib import Path


def default_db_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    path = root / "lumen" / "lumen.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _mean(values) -> float | None:
    nums = [float(value) for value in values if isinstance(value, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def downsample(points: list[dict], max_points: int = 700) -> list[dict]:
    count = len(points)
    if count <= max_points:
        return points
    output = []
    size = count / max_points
    for index in range(max_points):
        chunk = points[int(index * size) : int((index + 1) * size)]
        if not chunk:
            continue
        gpu_count = max((len(point.get("gpus") or []) for point in chunk), default=0)
        gpus = []
        for gpu_index in range(gpu_count):
            series = []
            for point in chunk:
                group = point.get("gpus") or []
                series.append(group[gpu_index] if gpu_index < len(group) else None)
            gpus.append(
                {
                    "util": _mean([item.get("util") if item else None for item in series]),
                    "vram": _mean([item.get("vram") if item else None for item in series]),
                    "power": _mean([item.get("power") if item else None for item in series]),
                    "temp": _mean([item.get("temp") if item else None for item in series]),
                    "power_limit": next(
                        (item.get("power_limit") for item in reversed(series) if item and item.get("power_limit")),
                        None,
                    ),
                }
            )
        output.append(
            {
                "ts": chunk[-1]["ts"],
                "cpu": _mean([point.get("cpu") for point in chunk]),
                "ram": _mean([point.get("ram") for point in chunk]),
                "gpus": gpus,
            }
        )
    return output


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._schema()

    def _schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                pid INTEGER,
                kind TEXT,
                proc_name TEXT,
                command TEXT,
                model TEXT,
                status TEXT,
                started_at REAL,
                ended_at REAL,
                proc_started_at REAL,
                peak_gpu REAL,
                peak_vram INTEGER,
                peak_power REAL,
                peak_temp REAL,
                avg_gpu REAL,
                sample_count INTEGER DEFAULT 0,
                gpu_samples INTEGER DEFAULT 0,
                gpu_name TEXT
            );
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                cpu REAL,
                ram_percent REAL,
                ram_used INTEGER,
                ram_total INTEGER,
                gpu_util REAL,
                vram_percent REAL,
                vram_used INTEGER,
                vram_total INTEGER,
                power_w REAL,
                temp_c REAL,
                disk_read REAL,
                disk_write REAL,
                gpus_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts);
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                run_id TEXT,
                type TEXT,
                data TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id, id);
            """
        )
        self.conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass

    def prune(self, days: int = 14) -> None:
        cutoff = time.time() - days * 86400
        with self._lock:
            self.conn.execute("DELETE FROM samples WHERE ts < ?", (cutoff,))
            self.conn.execute("DELETE FROM metrics WHERE ts < ?", (cutoff,))
            self.conn.commit()

    def open_run(self, proc: dict, ts: float | None = None, overwrite_model: bool = False) -> str:
        with self._lock:
            pid = int(proc["pid"])
            row = self.conn.execute(
                "SELECT id FROM runs WHERE pid=? AND status='running' ORDER BY started_at DESC LIMIT 1",
                (pid,),
            ).fetchone()
            if row:
                self._update(row["id"], proc, overwrite_model)
                self.conn.commit()
                return row["id"]
            run_id = uuid.uuid4().hex[:12]
            now = ts if ts is not None else time.time()
            self.conn.execute(
                """
                INSERT INTO runs (
                    id, pid, kind, proc_name, command, model, status, started_at, proc_started_at,
                    sample_count, gpu_samples
                ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, 0, 0)
                """,
                (
                    run_id,
                    pid,
                    proc.get("kind") or "running",
                    str(proc.get("name") or "")[:200],
                    str(proc.get("command") or "")[:4000],
                    (str(proc["model"])[:500] if proc.get("model") else None),
                    now,
                    float(proc.get("create_time") or now),
                ),
            )
            self.conn.commit()
            return run_id

    def update_run(self, run_id: str, proc: dict, overwrite_model: bool = False) -> None:
        with self._lock:
            self._update(run_id, proc, overwrite_model)
            self.conn.commit()

    def _update(self, run_id: str, proc: dict, overwrite_model: bool) -> None:
        row = self.conn.execute("SELECT model, command, proc_name FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            return
        model = row["model"]
        incoming = proc.get("model")
        if incoming and (overwrite_model or not model):
            model = str(incoming)[:500]
        command = str(proc["command"])[:4000] if proc.get("command") else row["command"]
        name = str(proc["name"])[:200] if proc.get("name") else row["proc_name"]
        kind = proc.get("kind")
        if kind:
            self.conn.execute(
                "UPDATE runs SET model=?, command=?, kind=?, proc_name=? WHERE id=?",
                (model, command, kind, name, run_id),
            )
        else:
            self.conn.execute(
                "UPDATE runs SET model=?, command=?, proc_name=? WHERE id=?",
                (model, command, name, run_id),
            )

    def close_run(self, run_id: str, ended_at: float | None = None) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE runs SET status='ended', ended_at=? WHERE id=? AND status='running'",
                (time.time() if ended_at is None else ended_at, run_id),
            )
            self.conn.commit()

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            return self._run(row) if row else None

    def open_runs(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM runs WHERE status='running' ORDER BY started_at").fetchall()
            return [self._run(row) for row in rows]

    def list_runs(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT * FROM runs
                ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [self._run(row) for row in rows]

    def count_runs(self) -> int:
        with self._lock:
            return int(self.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])

    def count_samples(self) -> int:
        with self._lock:
            return int(self.conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0])

    def add_sample(self, point: dict) -> None:
        gpus = point.get("gpus") or []
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO samples (
                    ts, cpu, ram_percent, ram_used, ram_total, gpu_util, vram_percent,
                    vram_used, vram_total, power_w, temp_c, disk_read, disk_write, gpus_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    point.get("ts"),
                    point.get("cpu"),
                    point.get("ram"),
                    point.get("ram_used"),
                    point.get("ram_total"),
                    point.get("gpu_util"),
                    point.get("vram_percent"),
                    point.get("vram_used"),
                    point.get("vram_total"),
                    point.get("power_w"),
                    point.get("temp_c"),
                    point.get("disk_read"),
                    point.get("disk_write"),
                    json.dumps(gpus, separators=(",", ":")),
                ),
            )
            gpu_util = point.get("gpu_util")
            rows = self.conn.execute(
                """
                SELECT id, peak_gpu, peak_vram, peak_power, peak_temp, avg_gpu,
                       sample_count, gpu_samples, gpu_name
                FROM runs WHERE status='running'
                """
            ).fetchall()
            for row in rows:
                gpu_samples = row["gpu_samples"] or 0
                average = row["avg_gpu"]
                if gpu_util is not None:
                    average = ((average or 0) * gpu_samples + float(gpu_util)) / (gpu_samples + 1)
                    gpu_samples += 1
                self.conn.execute(
                    """
                    UPDATE runs
                    SET sample_count=?, gpu_samples=?, avg_gpu=?, peak_gpu=?, peak_vram=?,
                        peak_power=?, peak_temp=?, gpu_name=?
                    WHERE id=?
                    """,
                    (
                        (row["sample_count"] or 0) + 1,
                        gpu_samples,
                        average,
                        _peak(row["peak_gpu"], gpu_util),
                        _peak(row["peak_vram"], point.get("vram_used")),
                        _peak(row["peak_power"], point.get("power_w")),
                        _peak(row["peak_temp"], point.get("temp_c")),
                        point.get("gpu_name") or row["gpu_name"],
                        row["id"],
                    ),
                )
            self.conn.commit()

    def add_metric(self, ts: float, run_id: str | None, kind: str, data: dict) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO metrics (ts, run_id, type, data) VALUES (?, ?, ?, ?)",
                (ts, run_id, kind, json.dumps(data, allow_nan=False, separators=(",", ":"))),
            )
            self.conn.commit()

    def metrics_for(self, run_id: str, limit: int = 2000) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM metrics WHERE run_id=? ORDER BY id DESC LIMIT ?",
                (run_id, limit),
            ).fetchall()
            return [self._metric(row) for row in reversed(rows)]

    def recent_metrics(self, limit: int = 8) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT m.ts, m.type, m.data, m.run_id, r.model, r.kind
                FROM metrics m
                LEFT JOIN runs r ON r.id = m.run_id
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            items = []
            for row in rows:
                item = self._metric(row)
                item["model"] = row["model"]
                item["kind"] = row["kind"]
                items.append(item)
            return items

    def samples_between(self, start: float, end: float, max_points: int = 700) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT ts, cpu, ram_percent, gpus_json
                FROM samples
                WHERE ts >= ? AND ts <= ?
                ORDER BY ts ASC
                """,
                (start, end),
            ).fetchall()
            points = []
            for row in rows:
                try:
                    gpus = json.loads(row["gpus_json"] or "[]")
                except json.JSONDecodeError:
                    gpus = []
                points.append({"ts": row["ts"], "cpu": row["cpu"], "ram": row["ram_percent"], "gpus": gpus})
        return downsample(points, max_points)

    @staticmethod
    def _run(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "pid": row["pid"],
            "kind": row["kind"],
            "proc_name": row["proc_name"],
            "command": row["command"],
            "model": row["model"],
            "status": row["status"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "proc_started_at": row["proc_started_at"],
            "peak_gpu": row["peak_gpu"],
            "peak_vram": row["peak_vram"],
            "peak_power": row["peak_power"],
            "peak_temp": row["peak_temp"],
            "avg_gpu": row["avg_gpu"],
            "sample_count": row["sample_count"],
            "gpu_name": row["gpu_name"],
        }

    @staticmethod
    def _metric(row: sqlite3.Row) -> dict:
        try:
            data = json.loads(row["data"] or "{}")
        except json.JSONDecodeError:
            data = {}
        return {"ts": row["ts"], "type": row["type"], "run_id": row["run_id"], "data": data}


def _peak(old, new):
    if new is None:
        return old
    if old is None:
        return new
    return max(old, new)
