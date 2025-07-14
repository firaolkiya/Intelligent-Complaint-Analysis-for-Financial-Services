import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import pickle
import re
from tqdm import tqdm

def chunk_text(text, chunk_size=256, chunk_overlap=32):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        if i + chunk_size >= len(words):
            break
        i += chunk_size - chunk_overlap
    return chunks

def main():
    # Load cleaned data
    df = pd.read_csv('data/filtered_complaints.csv')
    print(f'Loaded {len(df)} cleaned complaints.')

    # Chunking
    chunk_size = 256
    chunk_overlap = 32
    all_chunks = []
    metadata = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        chunks = chunk_text(row['processed_narrative'], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({
                'complaint_id': idx,
                'product': row['Product'],
                'original_narrative': row['Consumer complaint narrative'][:200],
            })
    print(f'Total text chunks: {len(all_chunks)}')

    # Embedding
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    print(f'Loaded embedding model: {model_name}')
    embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype('float32')
    print(f'Embeddings shape: {embeddings.shape}')

    # FAISS Index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    index.add(embeddings)
    print(f'FAISS index created and {index.ntotal} vectors added.')

    # Save index and metadata
    os.makedirs('vector_store', exist_ok=True)
    faiss.write_index(index, 'vector_store/complaints_faiss.index')
    with open('vector_store/complaints_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print('Vector store and metadata saved in vector_store/.')

if __name__ == '__main__':
    main() 