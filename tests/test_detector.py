from lumen.detector import classify_process, drop_launchers, extract_model
from lumen.integrations import parse_ollama_ps


def test_training_command_carries_model():
    item = classify_process(
        "python.exe",
        ["python", "train.py", "--model_name_or_path", "org/model"],
        12,
        10,
    )
    assert item["kind"] == "training"
    assert item["model"] == "org/model"


def test_llama_server_is_serving():
    item = classify_process(
        "llama-server.exe",
        ["llama-server", "-m", r"C:\models\Meta-Llama-3-8B-Q4.gguf", "-c", "4096"],
        4,
        10,
    )
    assert item["kind"] == "serving"
    assert item["model"].endswith(".gguf")


def test_vllm_module_is_serving():
    item = classify_process(
        "python",
        ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", "meta-llama/Meta-Llama-3-8B"],
        8,
        10,
    )
    assert item["kind"] == "serving"
    assert item["model"] == "meta-llama/Meta-Llama-3-8B"


def test_ignores_browser_and_self():
    assert classify_process("chrome.exe", ["chrome", "--type=renderer"], 3, 1) is None
    assert classify_process("lumen.exe", ["lumen", "--demo"], 3, 1) is None
    assert classify_process("python.exe", ["python", "-m", "lumen"], 3, 1) is None


def test_extract_model_skips_python_module_flag():
    command = ["python", "-m", "axolotl.cli.train", "--model", "org/model"]
    assert extract_model(command) == "org/model"


def test_launchers_are_hidden_when_a_child_is_present():
    kept = drop_launchers(
        [
            {"pid": 1, "ppid": 0, "name": "accelerate", "command": "accelerate launch train.py"},
            {"pid": 2, "ppid": 1, "name": "python", "command": "python train.py"},
        ]
    )
    assert [item["pid"] for item in kept] == [2]


def test_ollama_ps_parser():
    models = parse_ollama_ps({"models": [{"name": "llama3:8b", "size": 10, "size_vram": 8}]})
    assert models[0]["name"] == "llama3:8b"
    assert models[0]["vram"] == 8
