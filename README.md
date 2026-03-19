## Offline LLM Deployment

The project now supports local/offline LLM inference through `src/llm_core/llm_client.py`.

Supported providers:

- `LLM_PROVIDER=local`: any OpenAI-compatible local service, such as `vLLM`, `LM Studio`, or a custom `/v1/chat/completions` endpoint
- `LLM_PROVIDER=ollama`: local `Ollama` service
- If neither remote key is configured, `LOCAL_LLM_BASE_URL` or `OLLAMA_MODEL` can also trigger local mode automatically

### 1. OpenAI-compatible local service

PowerShell example:

```powershell
$env:LLM_PROVIDER="local"
$env:LOCAL_LLM_BASE_URL="http://127.0.0.1:8000/v1"
$env:LOCAL_LLM_MODEL="qwen2.5-7b-instruct"
$env:LOCAL_LLM_API_KEY="local"
python experiments/run_experiment.py
```

Expected endpoint:

- `POST {LOCAL_LLM_BASE_URL}/chat/completions`

### 2. Ollama local deployment

Start Ollama and pull a local model first:

```powershell
ollama pull qwen2.5:7b-instruct
```

Then run the experiment:

```powershell
$env:LLM_PROVIDER="ollama"
$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
$env:OLLAMA_MODEL="qwen2.5:7b-instruct"
python experiments/run_experiment.py
```

### Notes

- The LLM is only used to return JSON like `{"mode": "...", "reason": "..."}` for policy-mode selection.
- If the local model service is unavailable or returns invalid output, the system falls back to the built-in rule-based selector.
