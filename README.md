# Regulatory Q&A Copilot

[![ci](https://github.com/athulyakollipara-web/regulatory-qa-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/athulyakollipara-web/regulatory-qa-copilot/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue) ![aws](https://img.shields.io/badge/AWS-Bedrock%20%7C%20Guardrails%20%7C%20Lambda-orange)

Answers compliance questions about US consumer-banking regulations, where **every sentence cites the exact paragraph it comes from**, and **every figure is checked against the cited text**. If the regulations don't answer the question, it says so instead of guessing.

- *"How much can a customer lose if their debit card is stolen and they tell us within two business days?"*
- *"Can we charge overdraft fees on one-time debit card purchases if the customer never opted in?"*
- *"How many days do we have to notify an applicant after a credit decision?"*

The corpus is the real text of five regulations from the official [eCFR](https://www.ecfr.gov), downloaded fresh on every CI run:

| Regulation | 12 CFR Part | Covers |
|---|---|---|
| Regulation B | 1002 | Equal credit opportunity, adverse action notices |
| Regulation E | 1005 | Electronic fund transfers, debit cards, overdraft opt-in, remittances |
| Regulation P | 1016 | Privacy notices, opt-out of information sharing |
| Regulation DD | 1030 | Truth in Savings: account disclosures, overdraft fee reporting |
| Regulation CC | 229 | Funds availability, check holds, substitute checks |

## How it works

```mermaid
flowchart LR
    Q[Question] --> S[Screen<br/>Guardrails: prompt attacks, PII]
    S --> RW[Rewrite query<br/>into regulatory terms]
    RW --> R[Hybrid retrieval<br/>BM25 + Titan vectors, RRF]
    R --> RR[Rerank<br/>Bedrock cross-encoder]
    RR --> G{Enough support?}
    G -- no --> NF[Not found:<br/>no guessing]
    G -- yes --> A[Answer<br/>Claude, cite every sentence]
    A --> C{Citation check<br/>cited? retrieved? figures match?}
    C -- fail --> A
    C -- fails twice --> X[Quote the rule<br/>with citations]
    C -- pass --> GR{Guardrails<br/>contextual grounding}
    GR -- below threshold --> X
    GR -- pass --> OUT[Cited answer<br/>+ links to eCFR]
```

| Step | What it does | Tech |
|---|---|---|
| Corpus | Downloads the regulations as XML and labels every paragraph with its full citation (`1005.6(b)(2)`), handling the `(a)` / `(1)` / `(i)` / `(A)` nesting and the letter-vs-roman ambiguity of `(i)` | eCFR API, Python |
| Chunking | One chunk per top-level paragraph with its sub-paragraphs, so a whole rule stays together but each sub-paragraph stays citable | |
| Retrieval | BM25 for exact legal terms + vector search for plain-English paraphrases, fused with Reciprocal Rank Fusion | Titan Text Embeddings V2 (hashing embedder offline), OpenSearch-ready |
| Reranking | A cross-encoder reads question and candidate together to order the top 5 | Bedrock Rerank (lexical reranker offline) |
| Support gate | Low relevance → "not found". The threshold is chosen by the evaluation sweep, not guessed | |
| Answer | Claude answers from numbered paragraphs only and must cite each sentence | Bedrock Converse, model routing with fallback, prompt caching |
| Citation check | Rejects uncited sentences, citations to paragraphs that weren't retrieved, and figures (`$50`, `60 days`) not present in the cited paragraph. Errors go back to the model once; if it still fails, the copilot quotes the rule instead | `regqa/answer/citations.py` |
| Grounding check | A second, model-based check scores the answer against the sources and the question | Bedrock Guardrails contextual grounding |
| Serving | REST API with one-click section lookup; Lambda behind an HTTP API with Cognito auth | FastAPI, Lambda, API Gateway, Cognito, AWS CDK |
| Evaluation | 50 questions (40 answerable with gold sections across all five regulations, 10 out of scope), logged to MLflow, run in CI | MLflow, GitHub Actions |

## Results

Each CI run downloads the current regulations and evaluates the **offline pipeline** (hashing embeddings, lexical reranker, extractive answers, no AWS). The results table is on the run's summary page: **Actions → latest `ci` run → Summary**.

<!-- RESULTS: replace this comment with the table from the latest CI run summary -->

`bedrock-eval` (Actions → Run workflow) evaluates the full AWS pipeline: Titan embeddings, Bedrock rerank, Claude answers and Guardrails grounding. It needs the AWS secrets described in [Deploy](#deploy-to-aws).

**Metrics:** retrieval recall@1 and @5 against the gold section · share of answers citing the gold section · citation-check pass rate · correct "not found" on out-of-scope questions · a threshold sweep for the support gate.

## Quickstart

```bash
git clone https://github.com/athulyakollipara-web/regulatory-qa-copilot.git && cd regulatory-qa-copilot
make setup && source .venv/bin/activate
make test      # 27 tests: parser, chunking, citation checker, agent retries/fallbacks, AWS request contracts, API, CDK
make corpus    # downloads Regs B, E, P, DD, CC from eCFR (about 1 minute)
make eval      # 50-question evaluation, offline
python -m regqa.cli ask "Can a bank send a debit card to someone who did not ask for one?" --trace
make serve     # http://localhost:8000/docs
```

Full AWS pipeline:

```bash
export REGQA_LLM=bedrock REGQA_RERANKER=bedrock REGQA_EMBEDDINGS=titan AWS_REGION=us-east-1 RERANK_REGION=us-west-2
python -m regqa.cli ask "How long does a bank have to investigate a debit card error?"
```

## Design decisions

- **Citations at paragraph level, not page level.** Compliance reviewers check the exact paragraph. The parser keeps a full label for every paragraph, so the answer can cite `§ 1005.6(b)(1)` and the checker can verify that paragraph was actually retrieved.
- **Check the figures, not just the citations.** The most damaging error in a regulation answer is a cited sentence with the wrong number ($500 instead of $50). The checker requires every figure in a sentence to appear in the paragraph it cites (`test_wrong_figure_is_caught`).
- **Two independent checks.** The deterministic citation check catches structural problems. Guardrails contextual grounding catches unsupported meaning. Either one failing means the copilot falls back to quoting the rule.
- **Prefer "not found" to a guess.** Out-of-scope questions (tax, securities, deposit insurance) are refused, and the gate threshold is tuned from the evaluation sweep.
- **Runs without AWS.** Hashing embeddings, a lexical reranker and an extractive answerer keep the whole pipeline testable in CI at no cost. The AWS components replace them through the same interfaces.
- **AWS request shapes are tested.** Rerank and Guardrails requests are validated against botocore's official API models (`tests/test_aws_contracts.py`), so a wrong parameter fails in CI, not in a paid run.
- **Fixture is clearly fake.** Unit tests use an invented "Part 9999" (`tests/fixtures/part_9999.xml`), so no test can be mistaken for real legal text.

## Deploy to AWS

```bash
npm install -g aws-cdk && make deploy
aws s3 cp data/corpus.jsonl s3://<CorpusBucket>/corpus.jsonl
```

The stack creates the corpus bucket, a Guardrail (prompt-attack filter, PII masking, contextual grounding at 0.75 grounding / 0.5 relevance), a container Lambda, and an HTTP API whose `/ask` route requires a Cognito token. For `bedrock-eval`, add repository secrets `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (IAM policy in `docs/bedrock-eval-iam-policy.json`) and optionally `BEDROCK_GUARDRAIL_ID`. Bedrock rerank models are only available in some regions, so set `RERANK_REGION` to one that has them enabled.

## Project structure

```
regqa/
  corpus/ecfr.py        eCFR download + XML parsing + paragraph labels
  corpus/chunks.py      paragraph-group chunks that keep citations
  retrieval/            BM25, embeddings, hybrid RRF, lexical + Bedrock rerankers
  answer/citations.py   citation and figure checker
  answer/extractive.py  offline answerer and safe fallback
  llm/                  Bedrock router, prompts
  agent/graph.py        LangGraph workflow
  safety.py             Guardrails input screen + contextual grounding
  api/app.py · cli.py · lambda_handler.py
eval/                   50 questions, evaluation + demo scripts
infra/                  CDK stack
tests/                  27 tests
```

Not legal advice: answers quote and summarise regulation text for research and must be reviewed by qualified compliance staff.
