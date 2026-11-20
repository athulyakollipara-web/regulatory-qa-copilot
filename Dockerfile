# Local demo: downloads the regulations at build time, then serves the API on port 8000.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY regqa ./regqa
RUN pip install --no-cache-dir -e . && python -m regqa.corpus.ecfr --out data/corpus.jsonl
EXPOSE 8000
CMD ["uvicorn", "regqa.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
