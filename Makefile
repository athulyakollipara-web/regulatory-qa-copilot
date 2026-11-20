.PHONY: setup corpus test eval eval-bedrock serve deploy

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -e ".[eval,dev]"

corpus:        ## download Regs B, E, P, DD, CC from the eCFR API (needs internet)
	python -m regqa.corpus.ecfr --out data/corpus.jsonl

test:
	python -m pytest

eval:          ## offline: hashing embeddings + lexical reranker + extractive answers
	python eval/run_eval.py

eval-bedrock:  ## Titan embeddings + Bedrock rerank + Claude answers + Guardrails grounding
	REGQA_LLM=bedrock REGQA_RERANKER=bedrock REGQA_EMBEDDINGS=titan python eval/run_eval.py

serve:
	uvicorn regqa.api.app:app --reload --port 8000

deploy:        ## needs AWS credentials, Docker and the CDK CLI; then upload data/corpus.jsonl to the CorpusBucket output
	cd infra && pip install -e "..[infra]" && cdk deploy
