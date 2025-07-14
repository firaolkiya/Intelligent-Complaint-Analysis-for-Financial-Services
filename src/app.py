import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from transformers import pipeline
import sys
import os

# Ensure src is in the path for imports if running from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline import load_vector_store, rag_answer, EMBEDDING_MODEL_NAME

st.set_page_config(page_title="CrediTrust Complaint Analyst", layout="centered")
st.title("CrediTrust Complaint Analyst")
st.write("Ask any question about customer complaints across Credit Cards, Personal Loans, BNPL, Savings Accounts, and Money Transfers.")

# Load models and vector store (cache for performance)
@st.cache_resource
def load_resources():
    index, metadata = load_vector_store('vector_store/complaints_faiss.index', 'vector_store/complaints_metadata.pkl')
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    llm_pipeline = pipeline('text-generation', model='gpt2')  # Replace with your preferred LLM
    return index, metadata, embed_model, llm_pipeline

index, metadata, embed_model, llm_pipeline = load_resources()

# Session state for chat
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Input box
with st.form(key='question_form', clear_on_submit=False):
    user_question = st.text_input("Your question:", value="", key="user_question")
    submit = st.form_submit_button("Ask")
    clear = st.form_submit_button("Clear")

if clear:
    st.session_state['history'] = []
    st.experimental_rerun()

if submit and user_question.strip():
    with st.spinner('Retrieving and generating answer...'):
        answer, retrieved = rag_answer(user_question, index, metadata, embed_model, llm_pipeline, k=5)
        sources = [r['original_narrative'] for r in retrieved]
        st.session_state['history'].append({
            'question': user_question,
            'answer': answer,
            'sources': sources
        })

# Display chat history
for entry in reversed(st.session_state['history']):
    st.markdown(f"**You:** {entry['question']}")
    st.markdown(f"**AI:** {entry['answer']}")
    with st.expander("Show sources used for this answer"):
        for i, src in enumerate(entry['sources'], 1):
            st.markdown(f"**Source {i}:**\n{src}") 