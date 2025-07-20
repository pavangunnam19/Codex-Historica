from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import uvicorn
import os
from PIL import Image

# Import AI libraries
import pytesseract
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# --- Configuration ---
# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
DB_PATH = "vector_db"
COLLECTION_NAME = "historical_docs"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = 't5-small'

# --- Model Cache ---
# This dictionary will hold our models in memory after they are loaded.
model_cache = {}

# --- FastAPI Lifespan Manager ---
# This function runs when the server starts and stops.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Load the AI models into the model_cache
    print("Server starting up...")
    print("Loading embedding model...")
    model_cache['embedding_model'] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print("Loading language model and tokenizer...")
    model_cache['tokenizer'] = T5Tokenizer.from_pretrained(LLM_MODEL_NAME)
    model_cache['llm_model'] = T5ForConditionalGeneration.from_pretrained(LLM_MODEL_NAME)
    
    print("All models loaded successfully.")
    
    yield # The server is now running
    
    # SHUTDOWN: Clear the cache
    print("Server shutting down...")
    model_cache.clear()

# Initialize the FastAPI app with our lifespan manager
app = FastAPI(title="Codex Historica AI", lifespan=lifespan)

# --- Reusable AI Logic Functions ---

def process_and_store_document(file_path: str):
    """Extracts text, chunks it, and stores embeddings in ChromaDB."""
    print(f"Processing {file_path}...")
    
    # 1. Extract Text
    try:
        img = Image.open(file_path)
        document_text = pytesseract.image_to_string(img, lang='eng')
    except Exception as e:
        print(f"Error during OCR: {e}")
        return
    
    # 2. Chunk Text
    paragraphs = document_text.split('\n\n')
    text_chunks = [p.strip().replace("\n", " ") for p in paragraphs if len(p.strip()) > 0]
    if not text_chunks:
        print("No text chunks found.")
        return

    # 3. Get the pre-loaded embedding model
    embedding_model = model_cache['embedding_model']
    embeddings = embedding_model.encode(text_chunks)
    
    # 4. Store in Vector DB
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(name=COLLECTION_NAME) # Clear old collection
    except ValueError:
        pass # Collection didn't exist, which is fine
        
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    chunk_ids = [f"chunk_{i}" for i in range(len(text_chunks))]
    
    collection.add(embeddings=embeddings, documents=text_chunks, ids=chunk_ids)
    print(f"Knowledge base updated for {file_path}. {len(text_chunks)} chunks stored.")

def query_document(question: str):
    """Searches the DB and generates an answer using pre-loaded models."""
    embedding_model = model_cache['embedding_model']
    tokenizer = model_cache['tokenizer']
    llm_model = model_cache['llm_model']
    
    # 1. Connect to DB and get collection
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except ValueError:
        return "Error: No document has been processed yet. Please upload a document first."

    # 2. Find relevant context
    question_embedding = embedding_model.encode([question])
    results = collection.query(query_embeddings=question_embedding, n_results=3)
    context = " ".join(results['documents'][0])

    # 3. Generate Answer
    prompt = f"question: {question} context: {context}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = llm_model.generate(input_ids, max_length=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

# --- HTML Frontend ---
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Codex Historica</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #3a7bd5;
            --secondary-color: #3a60d5;
            --background-color: #f4f7f6;
            --card-background: #ffffff;
            --text-color: #333;
            --subtle-text-color: #666;
            --border-color: #e0e0e0;
        }
        body {
            font-family: 'Poppins', sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            margin: 0;
            padding: 2rem;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            min-height: 100vh;
        }
        .container {
            width: 100%;
            max-width: 700px;
            background: var(--card-background);
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .header h1 {
            font-size: 2.5rem;
            font-weight: 600;
            color: var(--primary-color);
            margin: 0;
        }
        .header p {
            font-size: 1.1rem;
            color: var(--subtle-text-color);
        }
        .step {
            background-color: #fafafa;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        .step h3 {
            margin-top: 0;
            font-weight: 600;
            color: var(--text-color);
            border-left: 4px solid var(--primary-color);
            padding-left: 1rem;
        }
        input[type="file"] {
            display: none;
        }
        .file-label {
            display: block;
            text-align: center;
            padding: 1rem;
            border: 2px dashed var(--border-color);
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s, border-color 0.3s;
        }
        .file-label:hover {
            background-color: #f0f5ff;
            border-color: var(--primary-color);
        }
        #file-name {
            display: block;
            text-align: center;
            margin-top: 0.5rem;
            color: var(--subtle-text-color);
            font-style: italic;
        }
        input[type="text"] {
            width: 100%;
            padding: 0.8rem 1rem;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 1rem;
            box-sizing: border-box;
            transition: border-color 0.3s, box-shadow 0.3s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(58, 123, 213, 0.2);
        }
        button {
            width: 100%;
            padding: 0.9rem 1rem;
            margin-top: 1rem;
            font-size: 1rem;
            font-weight: 500;
            color: white;
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .status, .response {
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .status {
            background-color: #e7f3ff;
            border: 1px solid #b3d7ff;
        }
        .response {
            background-color: #e9f7ef;
            border: 1px solid #bce3c5;
        }
        .spinner {
            width: 20px;
            height: 20px;
            border: 3px solid rgba(0, 0, 0, 0.1);
            border-left-color: var(--primary-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .footer {
            text-align: center;
            margin-top: 2rem;
            color: var(--subtle-text-color);
            font-size: 0.9rem;
        }
        .footer a {
            color: var(--primary-color);
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Codex Historica</h1>
            <p>Your AI Research Assistant</p>
        </div>

        <div class="step">
            <h3>Step 1: Upload a Document Page</h3>
            <form id="uploadForm">
                <label for="fileInput" class="file-label">Click to select a document image</label>
                <input type="file" name="file" id="fileInput" accept="image/png, image/jpeg, image/jpg" required>
                <span id="file-name">No file selected</span>
                <button type="submit" id="uploadButton">Upload and Process</button>
            </form>
            <div class="status" id="upload-status" style="display:none;"></div>
        </div>

        <div class="step">
            <h3>Step 2: Ask a Question</h3>
            <form id="queryForm">
                <input type="text" id="questionInput" placeholder="e.g., What was the purpose of this letter?" required>
                <button type="submit" id="askButton">Ask AI</button>
            </form>
            <div class="response" id="response" style="display:none;"></div>
        </div>
        
        <div class="footer">
            <p>by Pavan Gunnam</p>
        </div>
    </div>

    <script>
        const uploadForm = document.getElementById('uploadForm');
        const fileInput = document.getElementById('fileInput');
        const fileNameSpan = document.getElementById('file-name');
        const uploadStatus = document.getElementById('upload-status');
        const uploadButton = document.getElementById('uploadButton');

        const queryForm = document.getElementById('queryForm');
        const questionInput = document.getElementById('questionInput');
        const responseDiv = document.getElementById('response');
        const askButton = document.getElementById('askButton');

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                fileNameSpan.textContent = fileInput.files[0].name;
            } else {
                fileNameSpan.textContent = 'No file selected';
            }
        });

        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (fileInput.files.length === 0) {
                uploadStatus.innerHTML = '<span>Please select a file.</span>';
                uploadStatus.style.display = 'flex';
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            uploadStatus.innerHTML = '<div class="spinner"></div><span>Uploading and processing...</span>';
            uploadStatus.style.display = 'flex';
            uploadButton.disabled = true;

            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                const result = await response.json();
                uploadStatus.innerHTML = `<span>${result.message}</span>`;
            } catch (error) {
                uploadStatus.innerHTML = '<span>An error occurred during upload.</span>';
            } finally {
                uploadButton.disabled = false;
            }
        });

        queryForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            responseDiv.innerHTML = '<div class="spinner"></div><span>Thinking...</span>';
            responseDiv.style.display = 'flex';
            askButton.disabled = true;

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: new URLSearchParams({ question: questionInput.value })
                });
                const result = await response.json();
                responseDiv.innerHTML = `<span><strong>Answer:</strong> ${result.answer}</span>`;
            } catch (error) {
                responseDiv.innerHTML = '<span>An error occurred while asking the AI.</span>';
            } finally {
                askButton.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    return html_content

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    process_and_store_document(temp_file_path)
    
    os.remove(temp_file_path)
    return {"message": f"Successfully processed {file.filename}. You can now ask questions."}

@app.post("/ask")
async def ask_ai(question: str = Form(...)):
    answer = query_document(question)
    return {"answer": answer}

# --- Main entry point ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
