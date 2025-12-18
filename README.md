# 📘 AmbedkarGPT

### Semantic Graph Retrieval-Augmented Question Answering over Ambedkar’s Writings

AmbedkarGPT is a **Semantic Graph-based Retrieval Augmented Generation (SemRAG)** system built on the writings of **Dr. B. R. Ambedkar**.
It answers conceptual and analytical questions by combining **knowledge graphs**, **community-aware retrieval**, and **local LLM inference**, ensuring **faithful and explainable answers grounded strictly in the source text**.

---

## ✨ Key Features

* 📚 **Book-Grounded QA** – Answers are generated exclusively from Ambedkar’s writings
* 🧠 **Semantic Graph RAG (SemRAG)** – Entity-level and community-level retrieval
* 🔗 **Knowledge Graph Construction** – Entities, relations, and co-occurrence edges
* 🌍 **Dual Retrieval Strategy**

  * **Local Graph RAG** (fine-grained evidence)
  * **Global Graph RAG** (thematic context)
* 🤖 **Local LLM Inference** using Ollama (no external APIs)
* 🖥️ **Gradio-based Q&A Interface** (not a chatbot)

---

## 📁 Project Structure (Simplified)

```
AmbedkarGPT/
├── data/
│   ├── pages/              # Page-wise book text
│   ├── processed/          # Chunks and embeddings
│   ├── graph/              # Knowledge graphs & communities
├── src/
│   ├── chunking/           # Semantic chunking logic
│   ├── graph/              # Graph construction & community detection
│   ├── retrieval/          # Local & Global Graph RAG
│   └── pipeline/
│       └── AmbedkarGPT.py  # End-to-end demo (run this)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <private-repo-url>
cd AmbedkarGPT
```

---

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install Ollama (LLM Backend)

AmbedkarGPT uses **Ollama** for local LLM inference.

Download: [https://ollama.com](https://ollama.com)

Pull a model:

```bash
ollama pull mistral
# or
ollama pull llama3
```

Start Ollama:

```bash
ollama serve
```

---

## ▶️ Running the Demo (Interview-Ready)

Run the complete end-to-end system:

```bash
python src/pipeline/AmbedkarGPT.py
```

This launches a **Gradio Question–Answer interface** in your browser.

### Input

* Conceptual or analytical questions related to Ambedkar’s writings

  * Example: *“Explain caste in relation to religion and society”*

### Output

* ✅ Final generated answer
* 🔍 Local Graph RAG evidence (entity-based)
* 🌍 Global Graph RAG evidence (community-based)

---

## 🧠 System Architecture Overview

1. **Text Processing**

   * Page-wise extraction and semantic chunking

2. **Embedding Generation**

   * Sentence-transformer embeddings for chunks and summaries

3. **Knowledge Graph Construction**

   * Nodes: entities
   * Edges: relations + co-occurrence

4. **Community Detection**

   * Thematic clustering of graph nodes

5. **Dual Retrieval (SemRAG)**

   * Local Graph RAG → precise evidence
   * Global Graph RAG → thematic context

6. **Answer Generation**

   * Local LLM with strict prompt grounding

---

## 🎯 Design Goals

* Prevent hallucinations
* Preserve author intent
* Enable explainable retrieval
* Support academic & exam-style questions
* Demonstrate SemRAG principles clearly

---

## 🧪 Notes

* All preprocessing outputs are precomputed and stored in `data/`
* The system can be extended to other books by re-running the pipeline
* No external APIs or cloud services are required

