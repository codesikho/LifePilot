# LifePilot — Supervisor + Swarm integration

This archive contains the current LifePilot Python project plus a focused, runnable
Hybrid Supervisor + Swarm implementation. The offline demo uses fake specialists
so you can run the architecture without Ollama or an internet connection.

## Requirements

- Python 3.11 recommended
- For the offline demo/tests: `pydantic` and `pytest`
- For the real hotel ReAct agent: `langchain-core`, `langchain-ollama`, `langgraph`,
  and a running Ollama server with the configured model pulled locally.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the offline Hybrid demo

```bash
python -m ai.scripts.run_hybrid_demo
```

This demonstrates: Supervisor creates a plan → Travel specialist runs → Travel
requests a handoff → Swarm Router validates it → Budget specialist runs → Supervisor
returns the result. All hotel/cost data in this demo is fictional.

## Run tests

```bash
pytest -q
```

## Run the Ollama-backed Supervisor example

Start Ollama and pull the model first, for example:

```bash
ollama pull qwen3:8b
export OLLAMA_MODEL=qwen3:8b
python -m ai.scripts.run_supervisor
```

The hotel data is mock data only. The real Travel specialist uses the local hotel
dataset and does not book rooms or make payments. Live provider integrations and a
Shopping specialist are not yet implemented in this project snapshot.

## Main integration files

- `ai/orchestration/agent_dispatcher.py`: common execution contract and implementations
- `ai/orchestration/swarm_router.py`: validates and executes agent handoffs
- `ai/orchestration/hybrid_orchestrator.py`: wires Supervisor and Swarm together
- `ai/agents/supervisor/supervisor.py`: plans, dispatches, and aggregates specialist results
- `ai/scripts/run_hybrid_demo.py`: offline end-to-end demonstration

## Run the real Ollama-backed Hybrid workflow

After installing `requirements.txt`, starting Ollama and pulling `qwen3:8b`, run:

```bash
export OLLAMA_MODEL=qwen3:8b
python -m ai.scripts.run_hybrid_ollama
```

This uses the real Supervisor planner and ReAct Travel specialist with the mock
hotel dataset. The demo is not a live booking integration. In this project snapshot,
Travel does not yet emit a real dynamic Budget handoff; the offline demo separately
exercises the handoff path end to end with deterministic specialists.
