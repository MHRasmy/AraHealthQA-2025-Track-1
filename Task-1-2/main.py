import argparse
import json
import os
from types import SimpleNamespace

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_groq import ChatGroq
from huggingface_hub import InferenceClient
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ── environment ─────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


# ── SILMA wrappers (mirrors notebook classes) ───────────────────────────────
MODEL_ID_SILMA = "silma-ai/SILMA-Kashif-2B-Instruct-v1.0"


def _to_chat_list(messages):
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    chat = []
    for role, content in messages:
        role = {"human": "user", "ai": "assistant"}.get(role, role)
        chat.append({"role": role, "content": content})
    return chat


class SilmaLocalChat:
    def __init__(self, model_id=MODEL_ID_SILMA, temperature=0.0, max_new_tokens=256, load_in_4bit=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        qcfg = BitsAndBytesConfig(load_in_4bit=True) if load_in_4bit else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
            quantization_config=qcfg,
        )
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def invoke(self, messages):
        chat = _to_chat_list(messages)
        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        in_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0, in_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return SimpleNamespace(content=text.strip())


class SilmaHFEndpoint:
    def __init__(self, model_id=MODEL_ID_SILMA, temperature=0.0, max_new_tokens=256, timeout=120, provider=None):
        self.client = InferenceClient(model=model_id, token=HF_TOKEN, timeout=timeout, provider=provider)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def invoke(self, messages):
        chat = _to_chat_list(messages)

        try:
            out = self.client.chat_completion(
                messages=[{"role": m["role"], "content": m["content"]} for m in chat],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            msg = out.choices[0].message
            text = (msg["content"] if isinstance(msg, dict) else msg.content) or ""
            return SimpleNamespace(content=text.strip())
        except Exception as e_chat:
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            text = self.client.text_generation(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                return_full_text=False,
            )
            return SimpleNamespace(content=text.strip())


# ── prompts & strategies ────────────────────────────────────────────────────
TASK_LABELS = {
    "task1": ["Information", "Direct_Guidance", "Emotional_Support"],
    "task2": ["Factual", "Actionable", "Supportive"]
}

DEFAULT_FEW_SHOT = {
    "task1": [
        {"text": "ما هي أعراض القلق الشديد؟", "label": "Information"},
        {"text": "كيف أبدأ علاج الرهاب الاجتماعي خطوة بخطوة؟", "label": "Direct_Guidance"},
        {"text": "أشعر بالإحباط بعد فقدان عملي، هل يمكنك مواساتي؟", "label": "Emotional_Support"},
    ],
    "task2": [
        {"text": "يجب التواصل مع مختص نفسي لتقييم الحالة بدقة.", "label": "Actionable"},
        {"text": "الاكتئاب اضطراب معروف ويمكن علاجه بطرق متعددة.", "label": "Factual"},
        {"text": "أقدر شعورك بالتعب، ولست وحدك في هذا.", "label": "Supportive"},
    ],
}


def load_examples(path, task):
    if not path:
        return DEFAULT_FEW_SHOT[task]
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f]
        return json.load(f)


def build_prompt(text, task, strategy, examples, adapter_path=None):
    labels = TASK_LABELS[task]
    header = (
        "أنت مساعد لتقييم واستهداف مهام AraHealthQA.\n"
        f"المهمة: {'تصنيف السؤال' if task == 'task1' else 'تصنيف نمط الإجابة'}.\n"
        f"الخيارات المتاحة: {', '.join(labels)}.\n"
        "أعد فقط التصنيف الأنسب مع جملة قصيرة تبرر الاختيار."
    )

    strategy_lines = {
        "zero-shot": "الاستراتيجية: تصنيف Zero-shot بدون أمثلة مساعدة.",
        "few-shot": "الاستراتيجية: تصنيف Few-shot مع أمثلة إرشادية ثابتة.",
        "few-shot-frozen": "الاستراتيجية: Few-shot مع عمود فقري مجمد واستخدام الأمثلة كما هي.",
        "peft": f"الاستراتيجية: PEFT مع محولات مهيأة ({adapter_path or 'default PEFT adapter'}).",
        "instruction-tuning": "الاستراتيجية: نموذج تم ضبطه تعليمياً على بيانات AraHealthQA.",
    }

    body = [header, strategy_lines[strategy]]

    if strategy in {"few-shot", "few-shot-frozen"}:
        shot_lines = []
        for ex in examples:
            shot_lines.append(f"مثال:\n- نص: {ex['text']}\n- التصنيف: {ex['label']}")
        body.append("\n".join(shot_lines))

    body.append(f"النص:\n{text}")
    body.append("أعد التصنيف المختصر ثم تبرير بجملة واحدة.")
    return "\n\n".join(body)


def wrap_messages(prompt, model_choice):
    if model_choice == "silma":
        return prompt
    return [("system", "التزم بالقواعد، وأعد مخرجات موجزة."), ("human", prompt)]


# ── runner ──────────────────────────────────────────────────────────────────
def load_model(model_choice, use_local_silma=False):
    if model_choice == "qwen":
        return ChatGroq(
            model="qwen/qwen3-32b",
            api_key=GROQ_API_KEY,
            temperature=0,
            max_tokens=256,
            reasoning_effort="none",
            reasoning_format="hidden",
            timeout=None,
            max_retries=2,
        )
    if use_local_silma:
        return SilmaLocalChat(MODEL_ID_SILMA, temperature=0.0, max_new_tokens=256, load_in_4bit=True)
    return SilmaHFEndpoint(MODEL_ID_SILMA, temperature=0.0, max_new_tokens=256)


def run(args):
    model = load_model(args.model, use_local_silma=args.use_local_silma)
    examples = load_examples(args.few_shot_file, args.task)

    df = pd.read_csv(args.data_path, sep="\t")
    text_col = args.text_column if args.text_column in df.columns else df.columns[0]

    outputs = []
    for text in tqdm(df[text_col], desc="Processing"):
        prompt = build_prompt(text, args.task, args.strategy, examples, adapter_path=args.adapter_path)
        messages = wrap_messages(prompt, args.model)
        resp = model.invoke(messages)
        outputs.append(getattr(resp, "content", "").strip())

    pd.DataFrame(outputs).to_csv(args.output_path, sep="\t", index=False, header=False)
    print(f"✅ Saved results to {args.output_path}")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["task1", "task2"], required=True, help="task1=Question Classification, task2=Answer Categorization")
    parser.add_argument("--model", choices=["qwen", "silma"], required=True, help="Model to use")
    parser.add_argument("--strategy", choices=["zero-shot", "few-shot", "few-shot-frozen", "peft", "instruction-tuning"], required=True)
    parser.add_argument("--data-path", required=True, help="Input TSV path")
    parser.add_argument("--output-path", default="predictions.tsv", help="Where to write predictions TSV")
    parser.add_argument("--few-shot-file", help="Optional JSONL/JSON with examples (keys: text, label)")
    parser.add_argument("--adapter-path", help="Optional PEFT adapter reference")
    parser.add_argument("--text-column", default="question", help="Name of text column; falls back to first column if missing")
    parser.add_argument("--use-local-silma", action="store_true", help="Use local SILMA weights instead of HF endpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = cli()
    run(args)
