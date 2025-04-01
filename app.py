from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings  # Correct import
from langchain_pinecone import Pinecone as LangchainPinecone  # Correct import for Langchain Pinecone
import pinecone  # Correct import for pinecone client
from langchain_community.vectorstores import Pinecone as PineconeVectorStore  # Correct import for PineconeVectorStore
import requests
import os
from dotenv import load_dotenv
from src.prompt import *

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Create an instance of Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Set Pinecone index name
index_name = "medicalbots"

# Check if index exists, and create it if not
# Check if index exists, and create it if not
existing_indexes = pc.list_indexes()
if index_name not in existing_indexes:
    print(f"Index '{index_name}' not found. Creating it now...")
    
    # Define the index spec (cloud provider and region)
    spec = pinecone.ServerlessSpec(
        cloud="aws",  # You can change this based on your needs (e.g., 'gcp', 'azure')
        region="us-east-1"  # Change this to a valid region
    )

    # Create the index
    try:
        pc.create_index(
            name=index_name,
            dimension=384,  # Ensure the dimension is correct for your embeddings
            metric="cosine",  # You can change this if needed (e.g., 'euclidean')
            spec=spec
        )
        print(f"Index '{index_name}' created successfully.")
    except pinecone.core.openapi.shared.exceptions.PineconeApiException as e:
        # Handle the case where the index already exists (409 Conflict)
        if e.status == 409:
            print(f"Index '{index_name}' already exists.")
        else:
            # Rethrow any other exceptions
            raise e
else:
    print(f"Index '{index_name}' already exists.")

pinecone_index = pc.Index(index_name)

# Download embeddings
embeddings = HuggingFaceEmbeddings()  # Correct initialization for HuggingFace embeddings

# Initialize LangChain Pinecone Vector Store
docsearch = PineconeVectorStore(
    index=pinecone_index,  # Pass the pinecone index object
    embedding=embeddings,
    text_key="text"  # Default text key
)

# Set up retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# Hugging Face Inference API Call
def hf_generate_response(prompt):
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 150, "temperature": 0.4}}  # Fixed max_new_tokens

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=40)

        if response.status_code == 200:
            json_response = response.json()
            if isinstance(json_response, list) and len(json_response) > 0:
                return json_response[0].get("generated_text", "Error: No response generated.")
            else:
                return "Error: Unexpected response format from API."
        elif response.status_code == 429:
            return "Error: Rate limit exceeded. Please wait and try again."
        else:
            return f"Error: API responded with status {response.status_code}."
    
    except requests.Timeout:
        return "Error: API request timed out. Try again later."
    except requests.RequestException as e:
        return f"Error: {str(e)}"


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)

    # Retrieve relevant documents from Pinecone
    retrieved_docs = retriever.invoke(msg)
    if not retrieved_docs:  # Handle case where no documents are found
        context = "I couldn't find relevant information. Please rephrase your question."
    else:
        context = " ".join([doc.page_content for doc in retrieved_docs])

    # Adjust response based on user input
    if msg.lower() in ["explain more", "tell me more", "continue", "go on"]:
        full_prompt = f"{context}\nAssistant: Continue explaining in more detail."
    else:
        full_prompt = f"Context: {context}\nUser: {msg}\nAssistant:"

    response_text = hf_generate_response(full_prompt)

    # Ensure clean response
    if "Assistant:" in response_text:
        response_text = response_text.split("Assistant:", 1)[-1].strip()
    else:
        response_text = response_text.strip()

    print("Response:", response_text)

    return response_text  # Send only the refined response to frontend


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 for render
    app.run(host="0.0.0.0", port=port, threaded=True)
