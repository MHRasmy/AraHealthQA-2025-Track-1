# Task 1 & 2 Runner (Question Classification & Answer Categorization)

`main.py` runs Task‑1 (Question Classification) and Task‑2 (Answer Categorization) for either Qwen or SILMA, with selectable strategies:
- Zero/few‑shot prompting
- Few‑shot learning with frozen backbone
- PEFT
- Instruction Tuning

## Setup
1) Create and activate a virtualenv.
2) Install requirements for your chosen model:
   - Qwen: `pip install -r requirements-qwen.txt`
   - SILMA: `pip install -r requirements-silma.txt`
3) Add a `.env` file beside `main.py`:
   ```
   GROQ_API_KEY=...
   GEMINI_API_KEY=...   # kept for parity with Task‑3 envs
   HF_TOKEN=...         # required for SILMA via HuggingFace Inference
   ```

## Inputs
- TSV file containing the text to classify. Default text column name is `question`; the script falls back to the first column if the name is absent.
- Optional few‑shot examples file (`--few-shot-file`) as JSONL or JSON with keys `text` and `label`.
- Optional PEFT adapter reference (`--adapter-path`).

## Commands
- Task‑1, Qwen, zero‑shot:
  ```
  python3 main.py \
    --task task1 \
    --model qwen \
    --strategy zero-shot \
    --data-path /path/to/input.tsv \
    --output-path task1_qwen.tsv
  ```
- Task‑2, SILMA, few‑shot (HF endpoint):
  ```
  python3 main.py \
    --task task2 \
    --model silma \
    --strategy few-shot \
    --data-path /path/to/input.tsv \
    --output-path task2_silma.tsv
  ```
- Task‑1, SILMA local weights with PEFT adapter note:
  ```
  python3 main.py \
    --task task1 \
    --model silma \
    --strategy peft \
    --data-path /path/to/input.tsv \
    --output-path task1_silma_local.tsv \
    --adapter-path /path/to/adapter \
    --use-local-silma
  ```

## Outputs
- Predictions are written as a header‑less TSV (single column) to `--output-path`.
