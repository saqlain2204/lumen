"""A scripted workstation so the dashboard can be seen before a GPU job exists."""

from __future__ import annotations

import math
import random

DEMO_PID = 21440


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class DemoSource:
    def __init__(self) -> None:
        self.step = 0
        self.born: float | None = None
        self.rng = random.Random(7)

    def tick(self, ts: float) -> tuple[dict, list[dict], dict | None]:
        if self.born is None:
            self.born = ts
        self.step += 1
        ramp = min(1.0, self.step / 28)
        wave = 0.55 + 0.45 * math.sin(self.step / 16.0)
        util = _clamp(18 + ramp * (68 * wave + 10 + 8 * math.sin(self.step / 5.0)), 4, 99)
        power = _clamp(70 + 340 * (0.5 + 0.5 * math.sin(self.step / 10.0 + 1.2)), 40, 440)
        temp = _clamp(44 + util * 0.36, 38, 90)
        gb = 8 + 12 * (1 - math.exp(-self.step / 36))
        mem_total = 24 * 1024**3
        mem_used = int(gb * 1024**3)
        cpu = _clamp(12 + ramp * (22 + 14 * (0.5 + 0.5 * math.sin(self.step / 11))), 4, 96)
        ram = 36 + 18 * (1 - math.exp(-self.step / 40))
        ram_total = 64 * 1024**3
        ram_used = int(ram_total * ram / 100)
        cores = []
        for index in range(32):
            wobble = abs(math.sin(index * 1.7 + self.step / 9))
            cores.append(round(_clamp(cpu * (0.45 + 0.7 * wobble), 3, 97), 1))
        write = 120e6 if self.step % 50 == 0 else 3.5e6
        read = 8e6 + 6e6 * abs(math.sin(self.step / 13))
        host = {
            "hostname": "workstation",
            "os": "Windows 11",
            "cpu": "AMD Ryzen 9 7950X 16-Core Processor",
            "cores": 16,
            "threads": 32,
            "boot_time": ts - 104400,
            "uptime_s": 104400,
            "gpu_driver": "560.94",
        }
        gpu = {
            "index": 0,
            "name": "NVIDIA GeForce RTX 4090",
            "uuid": "GPU-DEMO-4090",
            "util": round(util, 1),
            "mem_util": round(_clamp(20 + (gb / 24) * 70, 0, 100), 1),
            "mem_used": mem_used,
            "mem_total": mem_total,
            "temp": round(temp, 1),
            "power_w": round(power, 1),
            "power_limit_w": 450.0,
            "clock_sm": int(1200 + util * 14),
            "clock_mem": 10251,
            "fan": round(_clamp(28 + util * 0.45, 20, 100), 0),
            "pcie": "Gen4 x16",
            "processes": [{"pid": DEMO_PID, "name": "python.exe", "mem": mem_used}],
        }
        snap = {
            "host": host,
            "cpu": {"percent": round(cpu, 1), "per_core": cores, "freq_mhz": 5450},
            "memory": {
                "percent": round(ram, 1),
                "used": ram_used,
                "total": ram_total,
                "swap_percent": 3.0,
                "swap_used": int(0.03 * 8 * 1024**3),
                "swap_total": 8 * 1024**3,
            },
            "disk": {"read_bps": read, "write_bps": write},
            "net": {"recv_bps": 1.2e6, "sent_bps": 0.3e6},
            "gpus": [gpu],
            "gpu_note": None,
        }
        proc = {
            "pid": DEMO_PID,
            "ppid": None,
            "name": "python.exe",
            "command": "python train.py --model meta-llama/Meta-Llama-3-8B --batch-size 2 --seq-len 2048",
            "kind": "training",
            "model": "meta-llama/Meta-Llama-3-8B",
            "create_time": self.born,
        }
        metric = None
        if self.step % 2 == 0:
            progress = min(1.0, self.step / 600)
            loss = 2.6 * math.exp(-self.step / 90) + 0.12 + self.rng.uniform(-0.015, 0.015)
            metric = {
                "step": self.step,
                "loss": round(loss, 4),
                "lr": 2e-4 * (0.5 + 0.5 * math.cos(math.pi * progress)),
                "tokens_per_sec": round(1800 + util * 18, 1),
                "epoch": round(self.step / 50, 3),
            }
        return snap, [proc], metric
