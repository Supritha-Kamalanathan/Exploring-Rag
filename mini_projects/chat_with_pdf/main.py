import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Load a pre-trained GPT-2 model to generate responses
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load the corresponding tokenizer to convert text into tokens and back
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load a pre-trained embedding model to encode text into vectors
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper function to clean the extracted text
def clean_text(text):
    # Remove characters that are not alphanumeric or whitespace
    text = re.sub(r"[^\w\s]", "", text)

    # Tokenize the text into individual words
    tokens = word_tokenize(text)

    # Normalization (convert all tokens to lower case)
    cleaned_tokens = [token.lower() for token in tokens]

    # Remove stopwords to focus on the meaningful words
    stop_words = set(stopwords.words("english"))
    cleaned_tokens = [token for token in cleaned_tokens if token not in stop_words]

    # Lemmatize tokens (reduces every word to its base or root form)
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in cleaned_tokens]

    return " ".join(cleaned_tokens)

# Function to extract text from PDF, split it into chunks and gather metadata
def get_chunks(pdf, chunk_size=400, chunk_overlap=50):
    texts_info = []
    text = ""
    if pdf is not None:
        content = PdfReader(pdf)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        for page_no, page in enumerate(content.pages):
            page_text = page.extract_text()
            text += page_text

            # Metadata to be used for data analysis or optimization
            # - knowing the approx token count is helpful for models which have token limitation
            #
            # texts_info.append({
            #     "page_number": page_no,
            #     "page_char_count": len(text),
            #     "page_word_count": len(text.split()),
            #     "page_sentence_count": len(text.split(". ")),
            #     "page_token_count": len(text) / 4,
            #     "text": page_text,
            # })

        text = clean_text(text)
        chunks = splitter.split_text(text)
        
    return chunks

# Function to encode text chunks into embeddings using SentenceTransformer
def calculate_embeddings(chunks):
    embeddings = []
    
    for chunk in chunks:
        embedding = embedding_model.encode(chunk) 
        embeddings.append(embedding)
    
    return np.array(embeddings)

# Function to generate a response based on retrieved context and the user's query
def generate_response(retrieved_docs, query):
    text = " ".join(retrieved_docs)

    prompt = f"""Context: {text}\n\nBased on the context, please answer the following question in your own words: {query}\n\nAnswer:"""

    # Ensure pad_token_id is set to valid value
    gpt_tokenizer.pad_token_id = gpt_tokenizer.eos_token_id 
    # - pad_token_id: ensures sequences are of equal length
    # - eos_token_id: indicates end of a sentence

    inputs = gpt_tokenizer.encode(prompt, return_tensors="pt")

    # Create an attention mask that allows transformer model to focus only on the meaningful tokens while ignoring the others
    attention_mask = inputs.ne(gpt_tokenizer.pad_token_id)

    outputs = gpt_model.generate(
        inputs, 
        attention_mask=attention_mask, # Creates a mask where the model will focus only on the non-padding tokens and ignores padding
        max_length=400,                # Generate upto 800 tokens
        num_return_sequences=1,        # Return a single response
        do_sample=True,                # Use sampling for more varied response
        top_p=0.95,                    # Consider smallest set of tokens whose cumulative probability is >= top_p
        top_k=50,                      # Choose from the top 50 most probable tokens
        temperature=0.5
        )

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
                chunks = get_chunks(pdf)
            with st.spinner("Calculating embeddings..."):
                embeddings = calculate_embeddings(chunks)

            query_embedding = embedding_model.encode(query)

            with st.spinner("Retrieving relevant docs..."):
                similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
                top_indices = similarities.argsort()[-3:][::-1] # Retrieve indices of top 3 chunks
                retrieved_docs = [chunks[i] for i in top_indices]

            with st.spinner("Generating response..."):
                response = generate_response(retrieved_docs, query)

            st.write(response)
            st.write(retrieved_docs)

if __name__ == "__main__":
    main()
