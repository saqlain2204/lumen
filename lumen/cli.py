"""Command line entry for the local observatory."""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser

import uvicorn

from lumen import __version__
from lumen.server import create_app
from lumen.store import default_db_path


def _open(url: str) -> None:
    time.sleep(0.6)
    webbrowser.open(url)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lumen",
        description="Local observatory for model runs and machine metrics.",
    )
    parser.add_argument("--version", action="version", version=f"lumen {__version__}")
    parser.add_argument("--demo", action="store_true", help="Simulate a GPU and a training run")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--db", default=None, help="Path to the SQLite file")
    args = parser.parse_args(argv)

    db = args.db or str(default_db_path())
    url = f"http://{args.host}:{args.port}"
    print(f"Lumen {__version__}  {url}")
    if args.demo:
        print("Demo mode. Simulated RTX 4090 and a training run. No hardware is being read.")
    else:
        print("Watching this machine. Sessions are saved locally.")
    print(f"Database  {db}")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(f"Bound to {args.host}. The dashboard and ingest API are reachable on that interface.")
    if not args.no_browser and args.host in {"127.0.0.1", "localhost", "::1"}:
        threading.Thread(target=_open, args=(url,), daemon=True).start()

    uvicorn.run(
        create_app(demo=args.demo, db_path=db, start=True),
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
    )
