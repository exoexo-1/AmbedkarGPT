📌 AmbedkarGPT — Graph-based Retrieval Augmented Generation (SemRAG)

AmbedkarGPT is a graph-augmented question-answering system built over the writings of Dr. B. R. Ambedkar, implementing Semantic Graph RAG (SemRAG) principles using entity graphs, community detection, and multi-level retrieval.

⚙️ Setup Instructions
1. Clone the Repository
git clone <private-repo-url>
cd AmbedkarGPT

2. Create and Activate Virtual Environment
python -m venv venv


Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Install and Run Ollama (LLM Backend)

This project uses Ollama for local LLM inference.

Download Ollama: https://ollama.com

Pull a model (recommended):

ollama pull mistral
# or
ollama pull llama3


Ensure Ollama is running:

ollama serve

▶️ Running the Demo (Interview-Ready)

The complete end-to-end pipeline (retrieval + answer generation + UI) is available as a single script.

python src/pipeline/AmbedkarGPT.py


This launches a Gradio-based Question–Answer interface in your browser.

Input: Natural language question related to Ambedkar’s writings

Output:

Generated answer

Local Graph RAG evidence

Global Graph RAG evidence

📂 Data & Preprocessing Notes

The book is preprocessed and stored in data/pages/

Chunking, embeddings, graph construction, and community summaries are precomputed

No additional preprocessing is required to run the demo

🧪 Optional: Notebook-Based Analysis

For step-by-step experimentation and diagnostics, see:

Testing/Process.ipynb

🔁 Rebuilding the Pipeline (Optional)

If required, individual pipeline stages can be re-run:

Chunking: src/chunking/

Graph construction: src/graph/

Retrieval: src/retrieval/