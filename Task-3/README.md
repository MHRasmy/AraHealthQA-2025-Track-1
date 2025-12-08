# AraHealthQA Task‑3 Runner

Single script (`main.py`) that runs the Task‑3 question‑answering flow for either Qwen‑3 or SILMA, using your local Chroma vector DB.

## Setup
1) Python env: `python3 -m venv .venv && source .venv/bin/activate`
2) Install deps (pick the model you want to run):
   - Qwen: `pip install -r requirements-qwen.txt`
   - SILMA: `pip install -r requirements-silma.txt`
   - If you have CUDA and prefer, you can swap `faiss-cpu` for `faiss-gpu`.
3) Add a `.env` file alongside `main.py` with the keys the notebooks used:
   ```
   GROQ_API_KEY=...
   GEMINI_API_KEY=...
   SERPER_API_KEY=...
   HF_TOKEN=...      # only needed for SILMA via HuggingFace Inference
   ```

## Inputs you need
- `Subtask3_input_test.tsv` (questions).
- A local Chroma DB directory (the notebooks used a persisted Chroma index; point `--chroma-dir` to that path).

## Run commands
- Qwen:
  ```
  python3 main.py \
    --model qwen \
    --data-path /path/to/Subtask3_input_test.tsv \
    --chroma-dir /path/to/chroma_index \
    --output-path output_answers.tsv
  ```
- SILMA (HuggingFace Inference; add `--use-local-silma` to load weights locally):
  ```
  python3 main.py \
    --model silma \
    --data-path /path/to/Subtask3_input_test.tsv \
    --chroma-dir /path/to/chroma_index \
    --output-path silma_answers.tsv
  ```

## Notes
- The script keeps the notebook logic intact; only the vector DB source was adjusted to accept a local Chroma path instead of rebuilding from a PDF.
- Requests are throttled (4.2s sleep) and retried on rate limits via the existing tenacity decorator.***
