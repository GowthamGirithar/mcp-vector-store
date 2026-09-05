.PHONY: install install-eval run test test-unit test-integration eval-deterministic eval-all

# Server dependencies only.
install:
	pip install -r requirements.txt

# Eval dependencies (pulls in requirements.txt plus ragas/langchain).
install-eval:
	pip install -r requirements-eval.txt

run:
	python main.py

# Full pytest suite (unit + integration).
test:
	pytest

# Mocked, fast tests only — no real vector DB / embedding service.
test-unit:
	pytest tests/unit

# Tests against real (temp-directory) vector DB + embedding services.
test-integration:
	pytest tests/integration

# Deterministic metrics only (hit-rate/MRR/NDCG) — no OpenAI judge, no Ragas.
eval-deterministic:
	PYTHONPATH=src python -m tests.eval.run_eval --no-judge

# Full eval including Ragas context_precision/context_recall via OpenAI judge.
eval-all:
	PYTHONPATH=src python -m tests.eval.run_eval
