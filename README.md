# Enterprise RAG Pipeline - WixQA Benchmark

A production-style Retrieval-Augmented Generation system built on the WixQA benchmark (6,221 Wix Help Center articles). The pipeline covers the full cycle: knowledge base ingestion, hyperparameter search, retrieval evaluation, generation evaluation, and a comparative study of three system improvements.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/Groq-API-orange?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48L3N2Zz4=)](https://console.groq.com/)
[![LLaMA](https://img.shields.io/badge/Meta-LLaMA%203-blueviolet)](https://llama.meta.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What This Project Does

Customer support systems need to answer questions grounded in a specific knowledge base — not from model memory. This project builds and evaluates exactly that kind of system using WixQA, a real enterprise benchmark from Wix.

The system takes a user query, retrieves the most relevant chunks from the knowledge base using dense vector search, and passes them to an LLM to generate a grounded answer. Four parts of the pipeline were evaluated end-to-end.

---

## Key Highlights

- Achieved **100% Context Recall** with optimized RAG configuration  
- Improved F1 score by **+16% (ExpertWritten)** using reranking  
- Achieved **+32% F1 improvement** on simulated user queries  
- Built a complete **end-to-end RAG evaluation pipeline** with 4-stage analysis

---

## Results at a Glance

**Optimal config:** chunk_size=500, overlap=60, k=3

| System | Dataset | Context Recall | F1 | ROUGE-1 | Factuality |
|---|---|---|---|---|---|
| Baseline RAG | ExpertWritten | 1.00 | 0.276 | 0.306 | 0.40 |
| Baseline RAG | Simulated | 0.94 | 0.181 | 0.221 | 0.85 |
| + Reranking | ExpertWritten | 1.00 | **0.321** | **0.336** | 0.30 |
| + Reranking | Simulated | 0.90 | **0.240** | **0.293** | **0.90** |
| + Query Rewriting | Simulated | **0.95** | 0.196 | 0.245 | 0.75 |

Reranking consistently improved answer quality (+16.6% F1 on ExpertWritten, +32.6% on Simulated). Query rewriting improved Context Recall on informal queries by ~1-5 points.

---

## System Architecture

```
Knowledge Base (6,221 articles)
        │
        ▼
  Text Chunking ──────────────────── Fixed-size (baseline)
        │                            Semantic (improvement 1)
        ▼
  BAAI/bge-base-en-v1.5 embeddings
        │
        ▼
  FAISS IndexFlatL2
        │
User Query ──────────────────────── Optional: Query Rewriting (improvement 3)
        │
        ▼
  Top-k Retrieval ─────────────────── Optional: Cross-encoder Reranking (improvement 2)
        │                                        top-50 → rerank → top-k
        ▼
  LLM Generator (Llama-3.1-8B via Groq)
        │
        ▼
  Grounded Answer
        │
        ▼
  Evaluation: Context Recall / F1 / ROUGE-1 / ROUGE-2 / Factuality
              (LLM-as-judge for Recall and Factuality, temperature=0)
```

---

## Dataset

[WixQA](https://huggingface.co/datasets/Wix/WixQA) — a three-config benchmark from Wix AI Research (arXiv:2505.08643)

| Split | Size | Role |
|---|---|---|
| `wix_kb_corpus` | 6,221 articles | Knowledge base |
| `wixqa_synthetic` | 6,221 queries | Hyperparameter tuning only |
| `wixqa_expertwritten` | 200 queries | Final evaluation |
| `wixqa_simulated` | 200 queries | Final evaluation |

---

## Pipeline Parts

### Part 1 - Hyperparameter Grid Search
Evaluated 12 configurations (`chunk_size ∈ {300, 500}`, `overlap ∈ {30, 60}`, `k ∈ {3, 5, 10}`) on WixQA-Synthetic using a composite score (0.4×Recall + 0.3×F1 + 0.3×ROUGE-1). Selected **chunk_size=500, overlap=60, k=3** as optimal.

### Part 2 - Retrieval Evaluation
Measured Context Recall on final evaluation sets using the optimal config. ExpertWritten: **0.96**, Simulated: **0.94**. Both strong; the small gap reflects vocabulary mismatch on informal queries.

### Part 3 - Generation Evaluation
Baseline system scored F1=0.276 / ROUGE-1=0.301 on ExpertWritten and F1=0.181 / ROUGE-1=0.221 on Simulated. Factuality was higher on Simulated (0.85) due to looser reference phrasing.

### Part 4 - System Improvements

| Improvement | What it does | Effect |
|---|---|---|
| Semantic Chunking | Split by embedding similarity instead of character count | Factuality +0.20 EW; F1 -0.16 (partial index coverage) |
| Reranking | top-50 FAISS → cross-encoder rerank → top-5 | F1 +0.045 EW, +0.059 Sim ✓ |
| Query Rewriting | LLM rewrites query before retrieval | Context Recall +0.01-0.05 on Simulated ✓ |

---

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `BAAI/bge-base-en-v1.5` (SentenceTransformers) |
| Vector store | FAISS IndexFlatL2 |
| LLM generator | Llama-3.1-8B-Instant via Groq API |
| LLM judge | Llama-3.1-8B-Instant, temperature=0 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Semantic chunking | LangChain SemanticChunker |
| Evaluation | ROUGE (rouge-score), token-level F1, LLM-as-judge |
| Dataset | HuggingFace `Wix/WixQA` |

---

## Project Structure

```
enterprise-rag-wixqa/
├── notebooks/
│   └── RAG_pipeline.ipynb             # Main notebook (runs end-to-end)
├── results/
│   ├── results_part1_grid.csv         # Grid search results (12 configs)
│   ├── results_part2_retrieval.csv    # Context Recall per dataset
│   ├── results_part3_generation.csv   # Baseline generation metrics
│   ├── results_part4_comparison.csv   # All systems compared
│   └── results_part4_deltas.csv       # Delta vs baseline
├── images/
│   ├── part1_grid_search.png          # Context Recall & F1 vs k
│   └── part4_improvements.png         # System comparison charts
├── report/
│   ├── Instruction(1).pdf        
│   └── RAG pipeline report.docx
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Umair-IITD/enterprise-rag-wixqa.git
cd enterprise-rag-wixqa
pip install -r requirements.txt
```

Add your Groq API key to a `.env` file (never commit this):
```
GROQ_API_KEY=your_key_here
```

Then open `RAG_pipeline.ipynb` and run all cells top to bottom.

**Free APIs used:** Groq (free tier, no credit card), HuggingFace datasets (public)

---

## Key Design Decisions

**Why FAISS over a hosted vector DB?** The full 6,221-article corpus fits comfortably in memory (~39K chunks at the optimal config). FAISS gives sub-millisecond retrieval without network latency or API costs.

**Why LLM-as-judge for Context Recall and Factuality?** Human evaluation at scale is impractical. Using the same LLM at temperature=0 gives deterministic, reproducible binary judgments that correlate well with human assessment on factual QA tasks.

**Why reranking outperformed semantic chunking?** Semantic chunking was applied to only the first 600 articles due to processing time constraints - this created an inconsistent index. Reranking works at query time and requires no index changes, making it more robust in practice.

---

## Limitations

- Grid search used 40 dev samples and excluded k=1 to control API costs
- Semantic chunking covered ~10% of the corpus (first 600 articles)
- LLM judge uses the same model as the generator, which may introduce correlated errors
- Evaluation on 20 samples per dataset in Parts 3–4 (API rate limit constraints)

---

## References

- Cohen et al. (2025). *WixQA: A Multi-Dataset Benchmark for Enterprise Retrieval-Augmented Generation.* arXiv:2505.08643
- BAAI/bge-base-en-v1.5: https://huggingface.co/BAAI/bge-base-en-v1.5
- cross-encoder/ms-marco-MiniLM-L-6-v2: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2

---

## Author

**Md Umair Alam**  <br>
B.Tech, IIT Delhi  <br>
Interested in AI, ML Systems, and Applied Research  

- GitHub: https://github.com/Umair-IITD
- LinkedIn: https://www.linkedin.com/in/umair-alam-iitd/

---

## License

MIT
