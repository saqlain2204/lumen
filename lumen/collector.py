"""One-shot snapshots of the local machine and any NVIDIA GPUs."""

from __future__ import annotations

import math
import os
import platform
import socket
import time

import psutil


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _num(value, digits: int = 1):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _mem_bytes(value):
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 0 or number > (1 << 48):
        return None
    return number


def cpu_name() -> str:
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                return str(name).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def host_info() -> dict:
    boot = psutil.boot_time()
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "cpu": cpu_name(),
        "cores": psutil.cpu_count(logical=False) or 0,
        "threads": psutil.cpu_count(logical=True) or 0,
        "boot_time": boot,
        "uptime_s": max(0.0, time.time() - boot),
        "gpu_driver": None,
    }


class Collector:
    def __init__(self) -> None:
        self.ready = False
        self.error: str | None = None
        self.driver: str | None = None
        self._pynvml = None
        self._last_disk = None
        self._last_net = None
        self._last_ts: float | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self.ready = True
            self.driver = _text(pynvml.nvmlSystemGetDriverVersion())
        except Exception:
            self.error = "NVIDIA management library is unavailable"

    def prime(self) -> None:
        psutil.cpu_percent(interval=0.1)
        psutil.cpu_percent(interval=None, percpu=True)
        self._last_disk = psutil.disk_io_counters()
        self._last_net = psutil.net_io_counters()
        self._last_ts = time.time()

    def snapshot(self) -> dict:
        now = time.time()
        cpu = _num(psutil.cpu_percent(interval=None), 1) or 0.0
        per_core = [_num(value, 1) or 0.0 for value in psutil.cpu_percent(interval=None, percpu=True)]
        freq = psutil.cpu_freq()
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        read_bps, write_bps = self._rates(psutil.disk_io_counters(), "disk", now)
        recv_bps, sent_bps = self._rates(psutil.net_io_counters(), "net", now)
        self._last_ts = now
        host = host_info()
        host["gpu_driver"] = self.driver
        gpus = self._gpus()
        note = None
        if not gpus:
            note = "No NVIDIA GPU detected. CPU, memory, and disk are still recording."
            if self.error:
                note = "GPU metrics are unavailable. CPU, memory, and disk are still recording."
        return {
            "host": host,
            "cpu": {"percent": cpu, "per_core": per_core, "freq_mhz": _num(freq.current, 0) if freq else None},
            "memory": {
                "percent": _num(virtual.percent, 1) or 0.0,
                "used": int(virtual.used),
                "total": int(virtual.total),
                "swap_percent": _num(swap.percent, 1) or 0.0,
                "swap_used": int(swap.used),
                "swap_total": int(swap.total),
            },
            "disk": {"read_bps": read_bps, "write_bps": write_bps},
            "net": {"recv_bps": recv_bps, "sent_bps": sent_bps},
            "gpus": gpus,
            "gpu_note": note,
        }

    def _rates(self, counters, kind: str, now: float) -> tuple[float, float]:
        previous = self._last_disk if kind == "disk" else self._last_net
        if kind == "disk":
            self._last_disk = counters
        else:
            self._last_net = counters
        if counters is None or previous is None or self._last_ts is None:
            return 0.0, 0.0
        dt = now - self._last_ts
        if dt < 0.5:
            return 0.0, 0.0
        if kind == "disk":
            return max(0.0, (counters.read_bytes - previous.read_bytes) / dt), max(
                0.0, (counters.write_bytes - previous.write_bytes) / dt
            )
        return max(0.0, (counters.bytes_recv - previous.bytes_recv) / dt), max(
            0.0, (counters.bytes_sent - previous.bytes_sent) / dt
        )

    def _gpus(self) -> list[dict]:
        if not self.ready or self._pynvml is None:
            return []
        pynvml = self._pynvml
        try:
            count = pynvml.nvmlDeviceGetCount()
        except Exception:
            return []
        devices = []
        for index in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            except Exception:
                continue
            devices.append(self._one_gpu(index, handle))
        return devices

    def _try(self, fn, default=None):
        try:
            return fn()
        except Exception:
            return default

    def _one_gpu(self, index: int, handle) -> dict:
        pynvml = self._pynvml
        util = self._try(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle))
        memory = self._try(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle))
        generation = self._try(lambda: pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle))
        width = self._try(lambda: pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle))
        pcie = f"Gen{generation} x{width}" if generation and width else None
        mem_used = _mem_bytes(getattr(memory, "used", None)) if memory else None
        mem_total = _mem_bytes(getattr(memory, "total", None)) if memory else None
        return {
            "index": index,
            "name": _text(self._try(lambda: pynvml.nvmlDeviceGetName(handle), f"GPU {index}")),
            "uuid": _text(self._try(lambda: pynvml.nvmlDeviceGetUUID(handle), "")),
            "util": _num(getattr(util, "gpu", None), 1) if util else None,
            "mem_util": _num(getattr(util, "memory", None), 1) if util else None,
            "mem_used": mem_used or 0,
            "mem_total": mem_total or 0,
            "temp": _num(self._try(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)), 1),
            "power_w": _num(self._mw(self._try(lambda: pynvml.nvmlDeviceGetPowerUsage(handle))), 1),
            "power_limit_w": _num(self._mw(self._try(lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(handle))), 1),
            "clock_sm": self._try(lambda: int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))),
            "clock_mem": self._try(lambda: int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))),
            "fan": _num(self._try(lambda: pynvml.nvmlDeviceGetFanSpeed(handle)), 0),
            "pcie": pcie,
            "processes": self._gpu_processes(handle),
        }

    @staticmethod
    def _mw(value):
        if value is None:
            return None
        return float(value) / 1000.0

    def _gpu_processes(self, handle) -> list[dict]:
        pynvml = self._pynvml
        found: dict[int, int | None] = {}
        for name in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            fn = getattr(pynvml, name, None)
            if fn is None:
                continue
            try:
                processes = fn(handle)
            except Exception:
                continue
            for proc in processes or []:
                pid = int(proc.pid)
                mem = _mem_bytes(getattr(proc, "usedGpuMemory", None))
                previous = found.get(pid)
                if previous is None or (mem or 0) > (previous or 0):
                    found[pid] = mem
        items = []
        for pid, mem in found.items():
            try:
                name = psutil.Process(pid).name()
            except (psutil.Error, OSError):
                name = str(pid)
            items.append({"pid": pid, "name": name, "mem": mem})
        items.sort(key=lambda item: item["mem"] or 0, reverse=True)
        return items
