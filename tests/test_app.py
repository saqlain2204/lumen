from fastapi.testclient import TestClient

from lumen.collector import Collector
from lumen.engine import Engine, Hub
from lumen.server import create_app
from lumen.store import Store


def test_collector_snapshot_has_machine_fields():
    snap = Collector().snapshot()
    assert "percent" in snap["cpu"]
    assert "used" in snap["memory"]
    assert isinstance(snap["gpus"], list)


def test_demo_engine_builds_a_full_session(tmp_path):
    store = Store(tmp_path / "demo.db")
    hub = Hub()
    engine = Engine(store, hub, demo=True)
    engine.start()
    try:
        payload = hub.latest()
        assert payload["demo"] is True
        assert len(payload["history"]) == 180
        assert "4090" in payload["gpus"][0]["name"]
        assert payload["runs"][0]["model"] == "meta-llama/Meta-Llama-3-8B"
        assert payload["runs"][0]["loss_tail"]
        assert payload["metrics"]
        assert store.count_samples() >= 180
        assert payload["runs"][0]["peak_vram"] > 0
    finally:
        engine.stop()
        store.close()


def test_claim_ingest_and_pages(tmp_path):
    app = create_app(db_path=tmp_path / "api.db", start=False)
    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "Lumen" in home.text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/app.css").status_code == 200
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True

        claimed = client.post(
            "/api/runs/claim",
            json={
                "pid": 42,
                "kind": "training",
                "model": "org/model",
                "name": "python",
                "command": "python train.py",
            },
        )
        assert claimed.status_code == 200
        run_id = claimed.json()["id"]
        ingested = client.post(
            "/api/ingest",
            json={"run_id": run_id, "type": "train", "metrics": {"loss": 1.5, "step": 3}},
        )
        assert ingested.status_code == 200
        detail = client.get(f"/api/runs/{run_id}")
        assert detail.status_code == 200
        body = detail.json()
        assert body["run"]["model"] == "org/model"
        assert body["metrics"][0]["data"]["loss"] == 1.5
        assert client.get("/api/runs/missing").status_code == 404


def test_model_claim_overwrites_blank_model(tmp_path):
    store = Store(tmp_path / "runs.db")
    run_id = store.open_run(
        {
            "pid": 9,
            "name": "python",
            "command": "python train.py",
            "kind": "training",
            "model": None,
            "create_time": 1,
        }
    )
    store.update_run(
        run_id,
        {"model": "org/model", "name": "python", "command": "python train.py", "kind": "training"},
        overwrite_model=True,
    )
    assert store.get_run(run_id)["model"] == "org/model"
    store.close()
