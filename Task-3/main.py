import argparse
import os
import time
from types import SimpleNamespace

import litellm
import pandas as pd
import torch
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    RetryError,
)
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── load environment variables ──────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# globals used by the copied notebook functions
llm = None
crew_llm = None


# ────────────────────── SILMA wrappers (from notebook) ──────────────────────
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
    def __init__(self, model_id=MODEL_ID_SILMA, temperature=0.7, max_new_tokens=500, load_in_4bit=False):
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
    def __init__(self, model_id=MODEL_ID_SILMA, temperature=0.0, max_new_tokens=500, timeout=120, provider=None):
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
            try:
                text = self.client.text_generation(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    return_full_text=False,
                )
                return SimpleNamespace(content=text.strip())
            except Exception as e_tg:
                raise RuntimeError(f"Both chat and text-generation calls failed: chat={e_chat} | text-gen={e_tg}")


# ────────────────────── shared notebook helpers ─────────────────────────────
def check_local_knowledge(query, context):
    """Router function to determine if we can answer from local knowledge"""
    prompt = '''Role: Question-Answering Assistant
Task: Determine whether the system can answer the user's question based on the provided text.
Instructions:
    - Analyze the text and identify if it contains the necessary information to answer the user's question.
    - Provide a clear and concise response indicating whether the system can answer the question or not.
    - Your response should include only a single word. Nothing else, no other text, information, header/footer.
Output Format:
    - Answer: Yes/No
Study the below examples and based on that, respond to the last question.
Examples:
    Input:
        Text: The dose of this medicine will be different for different patients. Follow your doctor's orders or the directions on the label. The following information includes only the average doses of this medicine. If your dose is different, do not change it unless your doctor tells you to do so. Children 10 to 17 years of age—At first, 0.5 mg once a day, in the morning or evening. Your doctor may adjust your dose as needed. However, the dose is usually not more than 6 mg per day.
        User Question: لدي بنت عندها تاخر في النمو العقلي وتاخد دوا واحد بس اسمو رسبيدرال ما بعرف قديش اعطيها حبه بليوم لاني بطلت اتابع فيها عند الدكتور من زمان وهسه عمرها ١٣ سنه وضعيفه
    Expected Output:
        Answer: Yes
    Input:
        Text: The episode is not attributable to the physiological effects of a substance or another medical condition. Note: Criteria A–C represent a major depressive episode. Note: Responses to a significant loss (e.g., bereavement, financial ruin, losses from a natural disaster, a serious medical illness or disability) may include the feelings of intense sadness, rumination about the loss, insomnia, poor appetite, and weight loss noted in Criterion A, which may resemble a depressive episode.
        User Question: لدي اكتئاب مفاجئ بعد أن تعرضت لصوت رعد مفاجئ وأمطار وأصبحت كئيية لمدة ثلاث ايام وأبكي بدون سبب هل هذه صدمة ام فقط فترة ثم قلت انه فقط اكتئاب المطر حتى أن اشرقت الشمس قد لازالت هذه الحالة معي لم اعرف السبب واحس كأنه هناك ضغط في صدري
    Expected Output:
        Answer: No
    Input:
        User Question: {query}
        Text: {text}
'''
    formatted_prompt = prompt.format(text=context, query=query)
    response = llm.invoke(formatted_prompt)
    return response.content.strip().lower() == "yes"


def setup_web_scraping_agent():
    """Setup the web scraping agent and related components"""
    search_tool = SerperDevTool()  # Tool for performing web searches
    scrape_website = ScrapeWebsiteTool()  # Tool for extracting data from websites

    # Define the web search agent
    web_search_agent = Agent(
        role="Expert Trusted‑Source Hunter",
        goal=(
          "For a given mental‑health question, locate ONE high‑quality, freely "
          "accessible webpage (Mayo Clinic, NIH, WHO, PubMed, etc.) that contains "
          "the *core context* needed to answer in the requested style "
          "(information / direct guidance / emotional support)."
        ),
        backstory=(
        "You are a specialised researcher who only relies on reputable health "
        "sites. You never choose forums, commercial blogs, or low‑authority pages."
        ),
        allow_delegation=False,
        verbose=True,
        llm=crew_llm
    )

    # Define the web scraping agent
    web_scraper_agent = Agent(
        role="Medical‑Context Extractor",
        goal=(
            "Download the page identified by the search agent and pull out the "
            "key paragraphs that supply the missing medical context. "
            "Return a clean JSON with title, url, and a concise Arabic summary."
        ),
        backstory=(
            "You have deep experience stripping boilerplate and focusing on medical "
            "facts, guidelines, or supportive language that matches the needed style."
        ),
        allow_delegation=False,
        verbose=True,
        llm=crew_llm,
    )

    # Define the web search task
    search_task = Task(
        description=(
            "Identify the most relevant web page or article for the topic: '{topic}'. "
            "Use all available tools to search for and provide a link to a web page "
            "that contains valuable information about the topic. Keep your response concise."
        ),
        expected_output=(
            "A concise summary of the most relevant web page or article for '{topic}', "
            "including the link to the source and key points from the content."
        ),
        tools=[search_tool],
        agent=web_search_agent,
    )

    # Define the web scraping task
    scraping_task = Task(
        description=(
            "Extract and analyze data from the given web page or website. Focus on the key sections "
            "that provide insights into the topic: '{topic}'. Use all available tools to retrieve the content, "
            "and summarize the key findings in a concise manner."
        ),
        expected_output=(
            "A detailed summary of the content from the given web page or website, highlighting the key insights "
            "and explaining their relevance to the topic: '{topic}'. Ensure clarity and conciseness."
        ),
        tools=[scrape_website],
        agent=web_scraper_agent,
    )

    # Define the crew to manage agents and tasks
    crew = Crew(
        agents=[web_search_agent, web_scraper_agent],
        tasks=[search_task, scraping_task],
        verbose=1,
        memory=False,
    )
    return crew


def get_web_content(query):
    """Get content from web scraping"""
    crew = setup_web_scraping_agent()
    result = crew.kickoff(inputs={"topic": query})
    return result.raw


def setup_vector_db(pdf_path):
    """Setup vector database from PDF"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db


def get_local_content(vector_db, query):
    """Get content from vector database"""
    docs = vector_db.similarity_search(query, k=5)
    return " ".join([doc.page_content for doc in docs])


# ────────────────────── Qwen notebook functions ─────────────────────────────
def generate_final_answer_qwen(context, query):
    """Generate final answer using LLM"""
    messages = [
        (
            "system",
            "You are **AraHealthQA Assistant**, a large‑language model helping Arabic speakers with mental‑health‑related questions. Use the provided context to answer the query accurately. Keep your answer ≤ 200 Arabic words unless user explicitly asked for detail.",
        ),
        ("system", f"Context: {context}"),
        ("human", query),
    ]
    response = llm.invoke(messages)
    return response.content


def process_query_qwen(query, vector_db, local_context):
    """Main function to process user query"""
    print(f"Processing query: {query}")

    can_answer_locally = check_local_knowledge(query, local_context)
    print(f"Can answer locally: {can_answer_locally}")

    if can_answer_locally:
        context = get_local_content(vector_db, query)
        print("Retrieved context from local documents")
    else:
        context = get_web_content(query)
        print("Retrieved context from web scraping")

    answer = generate_final_answer_qwen(context, query)
    return answer


@retry(
    wait=wait_exponential(multiplier=1, min=60, max=90),   # 60 s → 120 s → 240 s…
    stop=stop_after_attempt(15),                             # give up after ~15 min max
    retry=retry_if_exception_type(litellm.RateLimitError),
    reraise=True,
)
def safe_process_query_qwen(question, vector_db, local_ctx):
    """process_query with automatic back‑off"""
    return process_query_qwen(question, vector_db, local_ctx)


# ────────────────────── SILMA notebook functions ────────────────────────────
def generate_final_answer_silma(context, query):
    """Generate final answer using LLM"""
    messages = f"You are **AraHealthQA Assistant**, a large‑language model helping Arabic speakers with mental‑health‑related questions. Use the provided context to answer the query accurately. Keep your answer ≤ 200 Arabic words unless user explicitly asked for detail.\n\nContext:\n {context}\n\nQuery:\n{query}"
    response = llm.invoke(messages)
    return response.content


def process_query_silma(query, vector_db, local_context):
    """Main function to process user query"""
    print(f"Processing query: {query}")

    can_answer_locally = check_local_knowledge(query, local_context)
    print(f"Can answer locally: {can_answer_locally}")

    if can_answer_locally:
        context = get_local_content(vector_db, query)
        print("Retrieved context from local documents")
    else:
        context = get_web_content(query)
        print("Retrieved context from web scraping")

    answer = generate_final_answer_silma(context, query)
    return answer


@retry(
    wait=wait_exponential(multiplier=1, min=60, max=90),   # 60 s → 120 s → 240 s…
    stop=stop_after_attempt(15),                             # give up after ~15 min max
    retry=retry_if_exception_type(litellm.RateLimitError),
    reraise=True,
)
def safe_process_query_silma(question, vector_db, local_ctx):
    """process_query with automatic back‑off"""
    return process_query_silma(question, vector_db, local_ctx)


# ────────────────────── vector DB loader (Chroma) ───────────────────────────
def load_chroma_vector_db(chroma_dir):
    embedding_function = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-4B")
    return Chroma(persist_directory=chroma_dir, embedding_function=embedding_function)


# ────────────────────── runners ─────────────────────────────────────────────
def run_qwen(data_path, chroma_dir, output_path):
    global llm, crew_llm
    llm = ChatGroq(
        model="qwen/qwen3-32b",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
        max_tokens=500,
        reasoning_effort="none",
        reasoning_format="hidden",
        timeout=None,
        max_retries=2,
    )

    crew_llm = LLM(
        model="gemini/gemini-2.0-flash",
        api_key=os.environ["GEMINI_API_KEY"],
        max_tokens=500,
        temperature=0.7
    )

    print("🔄 Building vector DB …")
    vector_db = load_chroma_vector_db(chroma_dir)
    local_context = get_local_content(vector_db, "")

    df = pd.read_csv(data_path, sep="\t", header=None, names=["question"])

    answers = []
    for q in tqdm(df["question"], desc="Answering"):
        try:
            ans = safe_process_query_qwen(q, vector_db, local_context)
        except (litellm.RateLimitError, RetryError) as e:
            ans = f"[ERROR] Rate‑limit: {e}"
        except Exception as e:
            ans = f"[ERROR] {type(e).__name__}: {e}"
        answers.append(ans)

        time.sleep(4.2)

    pd.DataFrame(answers).to_csv(output_path, sep="\t", index=False, header=False)
    print(f"✅  All done – answers saved to {output_path}")


def run_silma(data_path, chroma_dir, output_path, use_local=False):
    global llm, crew_llm
    crew_llm = LLM(
        model="gemini/gemini-2.0-flash",
        api_key=os.environ["GEMINI_API_KEY"],
        max_tokens=500,
        temperature=0.7
    )

    if use_local:
        llm = SilmaLocalChat(MODEL_ID_SILMA, temperature=0.0, max_new_tokens=500, load_in_4bit=False)
    else:
        llm = SilmaHFEndpoint(MODEL_ID_SILMA, temperature=0.0, max_new_tokens=500)

    print("🔄 Building vector DB …")
    vector_db = load_chroma_vector_db(chroma_dir)
    local_context = get_local_content(vector_db, "")

    df = pd.read_csv(data_path, sep="\t", header=None, names=["question"])

    answers = []
    for q in tqdm(df["question"], desc="Answering"):
        try:
            ans = safe_process_query_silma(q, vector_db, local_context)
            print("answer: ", ans)
        except (litellm.RateLimitError, RetryError) as e:
            ans = f"[ERROR] Rate‑limit: {e}"
        except Exception as e:
            ans = f"[ERROR] {type(e).__name__}: {e}"
        answers.append(ans)

        time.sleep(4.2)

    pd.DataFrame(answers).to_csv(output_path, sep="\t", index=False, header=False)
    print(f"✅  All done – answers saved to {output_path}")


# ────────────────────── CLI entry ───────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen", "silma"], required=True, help="Model to run (qwen or silma)")
    parser.add_argument("--data-path", required=True, help="Path to Subtask3_input_test.tsv")
    parser.add_argument("--chroma-dir", required=True, help="Path to local Chroma vector DB")
    parser.add_argument("--output-path", default="output_answers.tsv", help="Where to write the TSV answers")
    parser.add_argument("--use-local-silma", action="store_true", help="Use local SILMA weights instead of HF endpoint")
    args = parser.parse_args()

    if args.model == "qwen":
        run_qwen(args.data_path, args.chroma_dir, args.output_path)
    else:
        run_silma(args.data_path, args.chroma_dir, args.output_path, use_local=args.use_local_silma)


if __name__ == "__main__":
    main()
