# Lumen

A local observatory for the machine and the model running on it.

When you train or serve a model on your own computer, Lumen records GPU load, VRAM, power, temperature, CPU, memory, and disk, and keeps each process as a session you can reopen. The dashboard is the instrument panel. The database never leaves the machine.

```bash
lumen --demo
```

That opens `http://127.0.0.1:8787` with a simulated RTX 4090 and a training run, so you can see the UI before a real job is running.

## Install

```bash
git clone https://github.com/saqlain2204/lumen.git
cd lumen
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
lumen
```

macOS and Linux:

```bash
source .venv/bin/activate
pip install -e .
lumen
```

Python 3.10 or newer. GPU numbers come from the NVIDIA driver. Machines without an NVIDIA GPU still record CPU, memory, disk, and process sessions.

## What it tracks

- CPU, per-core load, frequency, memory, swap, disk, and network
- NVIDIA utilization, VRAM, power draw and limit, temperature, fan, clocks, PCIe link, and the processes holding GPU memory
- A session for each detected training or inference process, with the machine timeline saved for that window
- Training and inference signals (loss, learning rate, tokens per second, time to first token) when a run reports them

Charts during a session are the whole machine. Two heavy jobs at once share one GPU curve.

## What it notices

Lumen scans the process list once a second and opens a session for:

- Ollama, llama.cpp (`llama-server`), vLLM, SGLang, LM Studio, koboldcpp, LocalAI, GPT4All
- Training commands such as `train.py`, Axolotl, Unsloth, TorchTune, DeepSpeed, `torchrun`, `accelerate launch`, LoRA, and QLoRA
- GGUF inference processes

Loaded Ollama models are read from `http://127.0.0.1:11434/api/ps`.

## Training and inference hooks

```python
from lumen import LumenCallback

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    callbacks=[LumenCallback()],
)
```

Any loop can log numbers:

```python
from lumen import log_metrics, log_inference

log_metrics({"loss": loss, "step": step, "lr": lr, "tokens_per_sec": tok_s}, type="train")

log_inference(
    model="llama3.1:8b",
    tokens_per_sec=54.2,
    ttft_ms=180,
    prompt_tokens=128,
    completion_tokens=64,
)
```

If Lumen is not running, these calls do nothing and training continues. The default address is `http://127.0.0.1:8787`. Point `LUMEN_URL` at another port if you change `--port`.

## Privacy

The server binds to `127.0.0.1`. Samples live in:

- Windows: `%LOCALAPPDATA%\lumen\lumen.db`
- macOS: `~/Library/Application Support/lumen/lumen.db`
- Linux: `~/.local/share/lumen/lumen.db`

Samples and metrics older than 14 days are deleted. Run rows are kept.

## Develop

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT.
