import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained GPT-2 model to generate responses
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load the corresponding tokenizer to convert text into tokens and back
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load a pre-trained embedding model to encode text into vectors
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper function to clean the extracted text
def format_text(text):
    formatted_text = text.replace("\n", ' ')
    return formatted_text

# Function to extract text from PDF, split it into chunks and gather metadata
def get_texts_info(pdf, chunk_size=1000, chunk_overlap=200):
    texts_info = []
    if pdf is not None:
        content = PdfReader(pdf)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        for page_no, page in enumerate(content.pages):
            text = format_text(page.extract_text())
            chunks = splitter.split_text(text)
            texts_info.append({
                "page_number": page_no,
                "page_char_count": len(text),
                "page_word_count": len(text.split()),
                "page_sentence_count": len(text.split(". ")),
                "page_token_count": len(text) / 4,
                "text": text,
                "chunks": chunks,
                "num_chunks": len(chunks)
            })
    return texts_info

# Function to encode text chunks into embeddings using SentenceTransformer
def calculate_embeddings(texts_info):
    chunks = []
    embeddings = []
    
    for i in texts_info:
        for chunk in i["chunks"]:
            chunks.append(chunk)
            embedding = embedding_model.encode(chunk) 
            embeddings.append(embedding)
    
    return chunks, np.array(embeddings)

# Function to generate a response based on retrieved context and the user's query
def generate_response(retrieved_docs, query):
    text = " ".join(retrieved_docs)

    prompt = f"""Context: {text}\n\nBased on the context, please answer the following question: {query}\n\nAnswer:"""

    inputs = gpt_tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt_model.generate(inputs, max_length=800, num_return_sequences=1, do_sample=True, top_p=0.95, top_k=50)

    response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean up the response by removing the original prompt
    response = response.replace(prompt, "").strip()

    return response

# Main function to handle streamlit UI and the overall pipeline
def main():
    st.set_page_config(
        page_title="Chat with your PDF",
        page_icon="🗪"
    )

    st.header("Chat with your PDF's 🗪")

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf:
        query = st.text_input("Your question: ")

        if query:
            with st.spinner("Getting texts info..."):
                texts_info = get_texts_info(pdf)
            with st.spinner("Calculating embeddings..."):
                chunks, embeddings = calculate_embeddings(texts_info)

            query_embedding = embedding_model.encode(query)

            with st.spinner("Retrieving relevant docs..."):
                similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
                top_indices = similarities.argsort()[-3:][::-1] # Retrieve indices of top 3 chunks
                retrieved_docs = [chunks[i] for i in top_indices]

            with st.spinner("Generating response..."):
                response = generate_response(retrieved_docs, query)

            st.write(response)

if __name__ == "__main__":
    main()
