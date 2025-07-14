# CrediTrust Financial Complaint Analysis RAG System

## Overview

This project implements an internal AI tool for CrediTrust Financial to transform raw, unstructured customer complaint data into actionable insights. Using Retrieval-Augmented Generation (RAG), the system enables product, support, and compliance teams to quickly understand customer pain points across Credit Cards, Personal Loans, BNPL, Savings Accounts, and Money Transfers.

**Key Features:**
- Fast, evidence-backed answers to plain-English questions about customer complaints
- Semantic search over complaint narratives using state-of-the-art embeddings
- Transparent, user-friendly web interface with source display for trust and verification

---

## Project Structure

```
.
├── data/
│   ├── complaints.csv                # Raw CFPB complaint data
│   └── filtered_complaints.csv       # Cleaned and filtered data
├── vector_store/
│   ├── complaints_faiss.index        # FAISS vector index
│   └── complaints_metadata.pkl       # Metadata for each vector
├── src/
│   ├── eda_preprocessing.py          # EDA and data cleaning script
│   ├── chunk_embed_index.py          # Chunking, embedding, and indexing script
│   ├── rag_pipeline.py               # RAG core logic and evaluation
│   └── app.py                        # Streamlit web application
├── notebooks/
│   └── eda_preprocessing.ipynb       # (Optional) EDA notebook
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Data

- Place the raw CFPB complaints CSV as `data/complaints.csv`.

### 5. Run EDA and Preprocessing

```bash
python src/eda_preprocessing.py
```
- This will generate `data/filtered_complaints.csv`.

### 6. Chunk, Embed, and Index

```bash
python src/chunk_embed_index.py
```
- This step may take a while. It will create the FAISS index and metadata in `vector_store/`.

### 7. Launch the Web App

```bash
cd src
streamlit run app.py
```
- Open the provided local URL in your browser.

---

## Usage

- **Ask a Question:** Type your question about customer complaints and click "Ask".
- **View Sources:** Expand the "Show sources used for this answer" section to see the complaint excerpts that informed the answer.
- **Clear Conversation:** Click "Clear" to reset the chat.

---

## Evaluation

- The system can be evaluated using the included `evaluate_rag` function in `rag_pipeline.py`.
- See the sample evaluation table in the report for qualitative analysis.

---

## Model & Design Choices

- **Chunking:** 256-word chunks with 32-word overlap for optimal context and retrieval.
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` for efficient, high-quality semantic search.
- **Vector Store:** FAISS for fast similarity search.
- **LLM:** Default is GPT-2 (can be swapped for a more advanced model).

---

## Screenshots

*(Include screenshots or a GIF of the Streamlit app here for your report.)*

---

## License

*Specify your license here (e.g., MIT, Apache 2.0, proprietary, etc.)*

---

## Acknowledgments

- Consumer Financial Protection Bureau (CFPB) for the open complaint dataset
- Hugging Face, FAISS, Streamlit, and Sentence Transformers communities

---

**For questions or support, contact:**  
*Your Name / Team / Email* 