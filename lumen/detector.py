"""Classify local processes that are training or serving a model."""

from __future__ import annotations

import re
import time
from pathlib import Path

import psutil

_TRAIN = re.compile(
    r"(fine[-_]?tune|unsloth|axolotl|torchtune|deepspeed|torchrun|qlora|\blora\b|\bsft\b|"
    r"pretrain|train\.py|sft\.py|finetune\.py)",
    re.IGNORECASE,
)
_HF_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
_PYTHON = {"python", "python3", "pythonw", "py"}
_LAUNCHERS = {"accelerate", "torchrun", "deepspeed"}
_SERVING_NAMES = {
    "ollama",
    "llama-server",
    "llama-cli",
    "koboldcpp",
    "localai",
    "text-generation-launcher",
    "lm studio",
    "lmstudio",
    "gpt4all",
    "tabbyapi",
}
_INFER_NAMES = {"comfyui", "invokeai", "fooocus"}
_SERVING_HINTS = (
    "llama-server",
    "vllm",
    "sglang",
    "koboldcpp",
    "text-generation-launcher",
    "text-generation-inference",
    "lm studio",
    "lmstudio",
    "ollama",
)
_INFER_HINTS = (
    "comfyui",
    "stable-diffusion",
    "invokeai",
    "fooocus",
    "text-generation-webui",
    "oobabooga",
    ".gguf",
)
_MODEL_FLAGS = {
    "--model",
    "--model-id",
    "--model_name",
    "--model_name_or_path",
    "--model-path",
    "--ckpt",
    "--checkpoint",
    "-m",
}


def _norm(name: str) -> str:
    text = (name or "").strip().lower()
    if text.endswith(".exe"):
        text = text[:-4]
    return text


def is_self(name: str, cmdline: list[str]) -> bool:
    if _norm(name) == "lumen":
        return True
    blob = " ".join(cmdline).lower().replace("\\", "/")
    return "-m lumen" in blob or "/lumen/cli.py" in blob or "/lumen/__main__.py" in blob


def _strip_python_module(cmdline: list[str]) -> list[str]:
    if not cmdline:
        return []
    stem = Path(cmdline[0]).stem.lower()
    if stem in _PYTHON and len(cmdline) >= 3 and cmdline[1] == "-m":
        return cmdline[3:]
    if stem in _PYTHON:
        return cmdline[1:]
    return cmdline


def extract_model(cmdline: list[str]) -> str | None:
    args = _strip_python_module(cmdline)
    for index, arg in enumerate(args):
        if arg.lower() in _MODEL_FLAGS and index + 1 < len(args):
            nxt = args[index + 1]
            if not nxt.startswith("-"):
                return nxt
    for arg in args:
        low = arg.lower()
        if low.endswith(".gguf") or low.endswith(".safetensors"):
            return Path(arg).name
    for arg in args:
        if _HF_ID.fullmatch(arg):
            return arg
    return None


def _kind(name: str, blob: str) -> str | None:
    normalized = _norm(name)
    low = blob.lower()
    if _TRAIN.search(low) or ("accelerate" in low and "launch" in low):
        return "training"
    if normalized in _SERVING_NAMES or normalized.startswith("ollama ") or any(hint in low for hint in _SERVING_HINTS):
        return "serving"
    if normalized in _INFER_NAMES or any(hint in low for hint in _INFER_HINTS):
        return "inference"
    return None


def classify_process(
    name: str,
    cmdline: list[str] | None,
    pid: int,
    create_time: float | None,
) -> dict | None:
    command = [str(part) for part in (cmdline or []) if part is not None]
    if is_self(name, command):
        return None
    blob = " ".join(command) if command else str(name or "")
    kind = _kind(name or "", blob)
    if kind is None:
        return None
    return {
        "pid": int(pid),
        "ppid": None,
        "name": name or "",
        "command": (" ".join(command)[:4000] or (name or "")),
        "kind": kind,
        "model": extract_model(command),
        "create_time": float(create_time or time.time()),
    }


def drop_launchers(items: list[dict]) -> list[dict]:
    parents = {item.get("ppid") for item in items if item.get("ppid")}
    kept = []
    for item in items:
        command = (item.get("command") or "").lower()
        is_parent = item.get("pid") in parents
        if is_parent and _norm(item.get("name") or "") in _LAUNCHERS:
            continue
        if is_parent and "accelerate" in command and "launch" in command:
            continue
        kept.append(item)
    return kept


def scan_processes(exclude_pids: set[int] | None = None) -> list[dict]:
    exclude = set(exclude_pids or ())
    found: list[dict] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            info = proc.info
            pid = info.get("pid")
            if pid is None or pid in exclude:
                continue
            item = classify_process(info.get("name") or "", info.get("cmdline") or [], pid, info.get("create_time"))
            if not item:
                continue
            try:
                item["ppid"] = proc.ppid()
            except (psutil.Error, OSError):
                item["ppid"] = None
            found.append(item)
        except (psutil.Error, OSError):
            continue
    return drop_launchers(found)
