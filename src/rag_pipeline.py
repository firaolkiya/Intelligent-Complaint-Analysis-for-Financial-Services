import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from transformers import pipeline

# Load embedding model (same as used for indexing)
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Prompt template
PROMPT_TEMPLATE = (
    "You are a financial analyst assistant for CrediTrust. "
    "Your task is to answer questions about customer complaints. "
    "Use the following retrieved complaint excerpts to formulate your answer. "
    "If the context doesn't contain the answer, state that you don't have enough information.\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Answer:"
)

def load_vector_store(index_path: str, metadata_path: str):
    index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    emb = model.encode([query], normalize_embeddings=True)
    return emb.astype('float32')

def retrieve(query: str, index, metadata, model, k=5) -> List[dict]:
    query_vec = embed_query(query, model)
    D, I = index.search(query_vec, k)
    results = []
    for idx in I[0]:
        results.append(metadata[idx])
    return results

def build_prompt(context_chunks: List[str], question: str) -> str:
    context = "\n---\n".join(context_chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)

def generate_answer(prompt: str, llm_pipeline) -> str:
    response = llm_pipeline(prompt, max_new_tokens=256, do_sample=False)
    if isinstance(response, list):
        return response[0]['generated_text']
    return response['generated_text']

def rag_answer(question: str, index, metadata, embed_model, llm_pipeline, k=5) -> Tuple[str, List[dict]]:
    retrieved = retrieve(question, index, metadata, embed_model, k)
    context_chunks = [r['original_narrative'] for r in retrieved]
    prompt = build_prompt(context_chunks, question)
    answer = generate_answer(prompt, llm_pipeline)
    return answer, retrieved

def evaluate_rag(questions: List[str], index, metadata, embed_model, llm_pipeline, k=5) -> pd.DataFrame:
    rows = []
    for q in questions:
        answer, retrieved = rag_answer(q, index, metadata, embed_model, llm_pipeline, k)
        row = {
            'Question': q,
            'Generated Answer': answer,
            'Retrieved Sources': '\n---\n'.join([r['original_narrative'][:200] for r in retrieved[:2]]),
            'Quality Score': '',  # To be filled in manually
            'Comments/Analysis': ''  # To be filled in manually
        }
        rows.append(row)
    return pd.DataFrame(rows)

if __name__ == '__main__':
    # Example usage (requires vector store and a local LLM pipeline)
    index, metadata = load_vector_store('vector_store/complaints_faiss.index', 'vector_store/complaints_metadata.pkl')
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # Example: use a local LLM (replace with your preferred model)
    llm_pipeline = pipeline('text-generation', model='gpt2')
    questions = [
        "Why are people unhappy with BNPL?",
        "What are the most common issues with credit cards?",
        "Are there complaints about money transfers being delayed?",
        "Do customers report fraud in personal loans?",
        "What problems do users face with savings accounts?"
    ]
    df_eval = evaluate_rag(questions, index, metadata, embed_model, llm_pipeline, k=5)
    df_eval.to_markdown('rag_evaluation.md', index=False)
    print(df_eval) 