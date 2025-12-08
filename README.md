# AraHealthQA 2025 – Track 1 (Tasks 1–3)

MINDWELL is a unified system that pairs Qwen-3 and SILMA across prompting, few-shot learning, PEFT, and instruction tuning, then anchors responses with a multi-agent RAG pipeline, bridging the gap in Arabic mental-health QA where prior work is small-scale, under-represented, and rarely couples multi-label classification with grounded generation for dialect- and terminology-diverse users.

Monorepo with runnable scripts for the three subtasks:
- **Task 1**: Question Classification (labels: information, guidance, emotional support)
- **Task 2**: Answer Categorization (factual, actionable, supportive tones)
- **Task 3**: Question Answering with retrieval (Qwen/SILMA + local Chroma DB)

## Structure
- `Task-1-2/` – unified runner for Task 1 & 2 with strategy switches (zero/few‑shot, frozen few‑shot, PEFT, instruction tuning). Model‑specific requirements files live here.
- `Task-3/` – QA runner for Qwen and SILMA over a local Chroma index; includes its own requirements files and README.

## Environment
- Create a virtualenv per task folder and install the matching requirements (Qwen or SILMA).
- Add a `.env` in each task folder with:
  ```
  GROQ_API_KEY=...
  GEMINI_API_KEY=...
  SERPER_API_KEY=...   # needed for Task‑3 web tools
  HF_TOKEN=...         # needed for SILMA HuggingFace Inference
  ```

## Running
- Task 1 & 2: see `Task-1-2/README.md` for strategy selection and commands.
- Task 3: see `Task-3/README.md` for QA over the local Chroma DB.

## Notes
- Each script writes header‑less TSV outputs for easy submission.
- The repository keeps model‑specific requirement files to keep Qwen/SILMA environments isolated.
