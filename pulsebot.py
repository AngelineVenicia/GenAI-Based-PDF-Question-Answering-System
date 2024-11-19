import streamlit as st
import base64
import warnings
import hashlib
import tempfile
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_huggingface import HuggingFaceEndpoint
from langchain_chroma import Chroma  # Updated import
from langchain_core.prompts import PromptTemplate  
from langchain_core.runnables import RunnableSequence  # Updated import
from langchain_community.document_loaders import PyPDFLoader

# Streamlit app configuration
st.set_page_config(
    page_title="PulseBot",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="auto"
)
st.markdown(
    "<h1 style='text-align: center;color:white;background-color:blue;'>PulseBot</h1>",
    unsafe_allow_html=True
)

st.subheader("Welcome!")

# Create a file uploader for PDF files
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Function to compute file hash
def compute_file_hash(file_path):
    """Compute a hash of the file to uniquely identify it."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Function to filter context based on query relevance
def filter_relevant_context(db, query):
    """Filter context to include only the most relevant parts of the document based on the query."""
    similar_docs = db.similarity_search(query, k=3)  # Return top 3 most relevant documents
    if similar_docs:
        relevant_context = " ".join([doc.page_content for doc in similar_docs])
        return relevant_context
    else:
        return "The document does not contain the information needed to answer this question."

# Main logic for handling file uploads and processing
if uploaded_file is not None:
    # Store the uploaded file in a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Compute a unique identifier for the uploaded PDF
    file_hash = compute_file_hash(temp_file_path)
   
    # Set up Chroma vector store directory path
    persist_directory = "C:/Users/320267920/OneDrive - Philips/Documents/streamlit/chromatb"
    embeddings_file_path = os.path.join(persist_directory, f"{file_hash}_embeddings")

    # Set up Hugging Face LLM and embeddings
    huggingfacehub_api_token = "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"
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    embeddings = HuggingFaceEmbeddings()  # Updated initialization

    # Initialize the LLM
    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=huggingfacehub_api_token,
        repo_id=repo_id,
        temperature=0.1,
        max_new_tokens=1500,
        top_p=0.1,
        top_k=5
    )

    # Check if the embeddings for the given file already exist
    if os.path.exists(embeddings_file_path):
        # Load existing embeddings
        db = Chroma(persist_directory=embeddings_file_path, embedding_function=embeddings)
        st.write("Loaded existing embeddings for the uploaded PDF.")
    else:
        # Load the PDF file
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load_and_split()

        # Split the pages into texts
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=0,
            separators=['\n\n', '\n', '(?=>\. )', ' ', '']
        )
        texts = text_splitter.split_documents(pages)

        # Create new Chroma vector store and persist the embeddings
        db = Chroma.from_documents(texts, embeddings, persist_directory=embeddings_file_path)

    # Create a text input for user questions
    user_input = st.text_input("Ask me anything")

    # Create a button for submitting the question
    if st.button("Get Response"):
        if user_input:
            # Filter relevant context based on the query
            relevant_context = filter_relevant_context(db, user_input)
            
            # Prepare the prompt with the filtered context
            template = """
            You are an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. Below is some information. 
            {context}

            Based on the above information only, answer the below question. 
            {question}
            """
            prompt = PromptTemplate.from_template(template)
            llm_chain = RunnableSequence(prompt | llm)  # Updated chaining

            # Get the response
            response = llm_chain.invoke({"context": relevant_context, "question": user_input})  # Updated method

            # Display the response
            st.write("**Response:**", response)
        else:
            st.write("Please enter a question.")

    # Optionally, clean up the temporary file if needed
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

# Display the footer with image
def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_path = "C:/Users/320267920/OneDrive - Philips/Documents/streamlit/logo.png"
image_base64 = get_image_as_base64(image_path)
url = "https://example.com"

footer_html = f"""
<style>
    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        padding: 10px;
    }}
</style>
<div class='footer'>
    <p>Developed with ❤️ by Angeline</p>
    <a href="{url}" target="_blank">
        <img src="data:image/png;base64,{image_base64}" alt="Clickable Image" width="80" height="50">
    </a>
</div>
"""

# Display the HTML
st.markdown(footer_html, unsafe_allow_html=True)
