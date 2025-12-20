import os
os.environ["MPLBACKEND"] = "Agg"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import gradio as gr
import ollama
# ---------- IMPORT YOUR EXISTING RETRIEVAL ----------
from src.retrieval.local_search import local_graph_rag_search
from src.retrieval.global_search import global_graph_rag_search

# ---------- CONFIG ----------
LLM_MODEL = "llama3.1:8b"     # or "mistral-7b" etc.
MAX_LOCAL_EVIDENCE = 5
MAX_GLOBAL_EVIDENCE = 10


# ---------- LLM ANSWER GENERATION ----------
def generate_answer(question, local_results, global_results):
    """
    Generate a grounded answer using Local + Global SemRAG evidence
    """

    local_context = "\n".join(
        f"- {r['text']}" for r in local_results[:MAX_LOCAL_EVIDENCE]
    )

    global_context = "\n".join(
        f"- {r['text']}" for r in global_results[:MAX_GLOBAL_EVIDENCE]
    )

    prompt = f"""
You are an expert academic analyst specializing in the works of Dr. B. R. Ambedkar.

Your task is to answer questions strictly based on the provided CONTEXT,
which is extracted from Ambedkar’s writings (e.g., *Castes in India*, *Annihilation of Caste*) but also use your own knowledge/ common sense.

INSTRUCTIONS:
- Use ONLY the information present in the CONTEXT.
- Do NOT introduce external facts, modern interpretations, or personal opinions.
- Do NOT quote page numbers explicitly unless they are already in the context.
- Focus on explaining arguments, critiques, and relationships between concepts,
  rather than giving dictionary-style definitions.
- When relevant, explicitly relate caste to religion, society, morality, law, or social structure,
  as discussed by Ambedkar.
- Write in a clear, academic, explainable tone.
- Avoid repetition; synthesize overlapping points into a coherent explanation.

QUESTION:
{question}

LOCAL CONTEXT (Entity-based, fine-grained evidence):
{local_context}

GLOBAL CONTEXT (Community-based, thematic evidence):
{global_context}

FINAL ANSWER:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )

    return response["message"]["content"].strip()


# ---------- SEMRAG PIPELINE ----------
def semrag_pipeline(question):
    if not question.strip():
        return "Please enter a question.", "", ""

    # ---- Retrieval ----
    local_out = local_graph_rag_search(question)
    global_out = global_graph_rag_search(question)

    # ---- LLM Generation ----
    answer = generate_answer(
        question,
        local_out["results"],
        global_out["results"]
    )

    # ---- Evidence Formatting ----
    local_evidence = "\n\n".join(
        f"• {r['text'][:350]} (pages {r['pages']})"
        for r in local_out["results"][:3]
    )

    global_evidence = "\n\n".join(
        f"• {r['text'][:350]} (pages {r['pages']})"
        for r in global_out["results"][:3]
    )

    return answer, local_evidence, global_evidence


# ---------- GRADIO UI ----------
with gr.Blocks(title="SemRAG Question Answering System") as demo:
    gr.Markdown("""
    # 📘 SemRAG-Based Question Answering System  
    **Local (Entity) + Global (Community) Graph Retrieval with LLM Answer Generation**

    Enter a question related to the document.
    """)

    question_box = gr.Textbox(
        label="Question",
        placeholder="e.g. What is Ambedkar’s critique of the caste system?",
        lines=2
    )

    answer_box = gr.Textbox(label="Final Answer", lines=10)
    local_box = gr.Textbox(label="Local Graph RAG Evidence", lines=8)
    global_box = gr.Textbox(label="Global Graph RAG Evidence", lines=8)

    submit_btn = gr.Button("Generate Answer")

    submit_btn.click(
        fn=semrag_pipeline,
        inputs=question_box,
        outputs=[answer_box, local_box, global_box]
    )

demo.launch()
