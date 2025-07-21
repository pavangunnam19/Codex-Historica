# main.py (Final Version: Stricter Prompting, Upgraded AI, Pro UI/UX)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import uvicorn
import os
from PIL import Image
import io
import asyncio

# Import AI and PDF libraries
import pytesseract
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import fitz  # PyMuPDF

# --- Configuration ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
DB_PATH = "vector_db"
COLLECTION_NAME = "historical_docs"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'google/flan-t5-large'

# --- Model Cache ---
model_cache = {}

# --- FastAPI Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Load AI models
    print("Server starting up...")
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model_cache['embedding_model'] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print(f"Loading language model and tokenizer: {LLM_MODEL_NAME}...")
    model_cache['tokenizer'] = T5Tokenizer.from_pretrained(LLM_MODEL_NAME)
    model_cache['llm_model'] = T5ForConditionalGeneration.from_pretrained(LLM_MODEL_NAME)
    
    print("All models loaded successfully. Application is ready.")
    yield
    # SHUTDOWN: Clear cache
    print("Server shutting down...")
    model_cache.clear()

app = FastAPI(title="Codex Historica AI", lifespan=lifespan)

# --- Synchronous AI Logic Functions ---

def process_and_store_document_sync(file_path: str, file_content: bytes):
    """Synchronous function to extract text, chunk it, and store embeddings."""
    print(f"Processing {file_path}...")
    full_document_text = ""
    
    try:
        if file_path.lower().endswith('.pdf'):
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap(dpi=300)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                full_document_text += pytesseract.image_to_string(img, lang='eng') + "\n\n"
            pdf_document.close()
        else:
            img = Image.open(io.BytesIO(file_content))
            full_document_text = pytesseract.image_to_string(img, lang='eng')
    except Exception as e:
        print(f"Error during file processing: {e}")
        raise RuntimeError(f"Failed to process file: {e}")

    paragraphs = full_document_text.split('\n\n')
    text_chunks = [p.strip().replace("\n", " ") for p in paragraphs if len(p.strip()) > 20]
    if not text_chunks:
        raise ValueError("No meaningful text could be extracted from the document.")

    embedding_model = model_cache['embedding_model']
    embeddings = embedding_model.encode(text_chunks)
    
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except ValueError:
        pass
        
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    chunk_ids = [f"chunk_{i}" for i in range(len(text_chunks))]
    collection.add(embeddings=embeddings, documents=text_chunks, ids=chunk_ids)
    print(f"Knowledge base updated. {len(text_chunks)} chunks stored.")
    return len(text_chunks)

def query_document_sync(question: str):
    """Synchronous function to search the DB and generate an answer."""
    embedding_model = model_cache['embedding_model']
    tokenizer = model_cache['tokenizer']
    llm_model = model_cache['llm_model']
    
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except ValueError:
        raise FileNotFoundError("No document has been processed yet. Please upload a document first.")

    question_embedding = embedding_model.encode([question])
    results = collection.query(query_embeddings=question_embedding, n_results=5)
    
    if not results['documents'] or not results['documents'][0]:
        return "I could not find any relevant information in the document to answer your question."
        
    context = " ".join(results['documents'][0])
    
    # --- FINAL, STRICTER PROMPT ---
    prompt = f"""
    You are a helpful AI assistant. Your task is to answer a question based *only* on the provided context document.

    Follow these steps:
    1.  Read the context carefully.
    2.  Identify the specific sentences or phrases within the context that directly answer the question.
    3.  Synthesize these findings into a concise and clear answer.
    4.  If the context does not contain the information needed to answer the question, you MUST state: "The provided document does not contain information on this topic." Do not use any external knowledge.

    Context:
    ---
    {context}
    ---

    Question: {question}

    Answer:
    """
    
    input_ids = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).input_ids
    outputs = llm_model.generate(input_ids, max_length=256, num_beams=5, early_stopping=True)
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
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0B0F19;
            --card-bg-color: rgba(255, 255, 255, 0.05);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-color: #E0E0E0;
            --subtle-text-color: #8892b0;
            --primary-color: #5E72EB;
            --primary-hover-color: #788bff;
            --success-color: #33d6a6;
            --error-color: #ff5e5e;
        }
        @keyframes aurora {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 2rem;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            overflow: hidden;
            position: relative;
        }
        body::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(125deg, #0d324d, #7f5a83, #1a1a2e, #16213e);
            background-size: 400% 400%;
            animation: aurora 15s ease infinite;
            filter: blur(50px);
            opacity: 0.3;
            z-index: -1;
        }
        .main-container {
            width: 100%;
            max-width: 1200px;
            height: 90vh;
            max-height: 800px;
            display: flex;
            gap: 2rem;
            background: var(--card-bg-color);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }
        .left-panel, .right-panel { display: flex; flex-direction: column; }
        .left-panel { flex: 1; min-width: 300px; }
        .right-panel { flex: 2; background: rgba(0,0,0,0.1); border-radius: 16px; }
        .header h1 { font-size: 1.8rem; font-weight: 700; margin: 0; }
        .header p { color: var(--subtle-text-color); margin-top: 0.5rem; }
        .upload-box { margin-top: 2rem; }
        .file-label { display: flex; flex-direction: column; justify-content: center; align-items: center; height: 150px; border: 2px dashed var(--border-color); border-radius: 12px; cursor: pointer; transition: all 0.3s; }
        .file-label:hover { border-color: var(--primary-color); background-color: rgba(94, 114, 235, 0.1); }
        .file-label svg { width: 40px; height: 40px; color: var(--subtle-text-color); margin-bottom: 1rem; }
        .status-box { margin-top: 1rem; padding: 0.8rem; background: rgba(0,0,0,0.2); border-radius: 8px; font-size: 0.9rem; }
        button { font-family: 'Inter', sans-serif; font-size: 1rem; font-weight: 600; color: white; background: var(--primary-color); border: none; border-radius: 8px; padding: 0.8rem 1rem; cursor: pointer; transition: background-color 0.3s; display: flex; justify-content: center; align-items: center; gap: 0.5rem; }
        button:hover:not(:disabled) { background: var(--primary-hover-color); }
        button:disabled { background: #555; cursor: not-allowed; }
        .chat-window { flex-grow: 1; padding: 1.5rem; overflow-y: auto; display: flex; flex-direction: column-reverse; }
        .chat-messages { display: flex; flex-direction: column; gap: 1rem; }
        .message { display: flex; gap: 0.8rem; max-width: 85%; }
        .message .content { padding: 0.8rem 1rem; border-radius: 12px; line-height: 1.5; }
        .message.user { align-self: flex-end; }
        .message.user .content { background: var(--primary-color); color: white; border-bottom-right-radius: 4px; }
        .message.ai { align-self: flex-start; }
        .message.ai .content { background: rgba(255, 255, 255, 0.1); border-bottom-left-radius: 4px; }
        .message.ai .avatar { background: #2c3e50; }
        .avatar { width: 40px; height: 40px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-weight: 600; flex-shrink: 0; }
        .chat-input-area { padding: 1.5rem; border-top: 1px solid var(--border-color); }
        .chat-input-form { display: flex; gap: 1rem; }
        input[type="text"] { flex-grow: 1; background: rgba(0,0,0,0.2); border: 1px solid var(--border-color); border-radius: 8px; padding: 0.8rem 1rem; color: var(--text-color); font-size: 1rem; }
        input[type="text"]:focus { outline: none; border-color: var(--primary-color); }
        .chat-input-form button { width: 50px; height: 50px; padding: 0; }
        .footer { text-align: center; margin-top: auto; padding-top: 1rem; color: var(--subtle-text-color); font-size: 0.8rem; }
        .toast { position: fixed; bottom: 2rem; left: 50%; transform: translateX(-50%); padding: 0.8rem 1.5rem; border-radius: 8px; color: white; font-weight: 500; box-shadow: 0 4px 20px rgba(0,0,0,0.3); z-index: 1000; opacity: 0; transition: opacity 0.5s, transform 0.5s; }
        .toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
        .toast.success { background: var(--success-color); }
        .toast.error { background: var(--error-color); }
        @media (max-width: 768px) {
            body { padding: 1rem; }
            .main-container { flex-direction: column; height: auto; max-height: none; }
            .right-panel { min-height: 400px; }
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="left-panel">
            <div class="header">
                <h1>Codex Historica</h1>
                <p>Your AI Research Assistant</p>
            </div>
            <div class="upload-box">
                <form id="uploadForm">
                    <label for="fileInput" class="file-label">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                        <span>Upload a Document</span>
                    </label>
                    <input type="file" id="fileInput" accept="image/png, image/jpeg, application/pdf" required>
                    <div class="status-box" id="status-box">
                        <strong>Status:</strong> <span id="doc-status">No document loaded.</span>
                    </div>
                    <button type="submit" id="uploadButton" style="margin-top: 1rem;">Process Document</button>
                </form>
            </div>
            <div class="footer">
                <p>by Pavan Gunnam</p>
            </div>
        </div>
        <div class="right-panel">
            <div class="chat-window" id="chat-window">
                <div class="chat-messages" id="chat-messages">
                    <div class="message ai">
                        <div class="avatar">AI</div>
                        <div class="content">Hello! Please upload a document to begin.</div>
                    </div>
                </div>
            </div>
            <div class="chat-input-area">
                <form class="chat-input-form" id="queryForm">
                    <input type="text" id="questionInput" placeholder="Ask a question..." required disabled>
                    <button type="submit" id="askButton" disabled>
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                    </button>
                </form>
            </div>
        </div>
    </div>
    <div id="toast" class="toast"></div>

    <script>
        const uploadForm = document.getElementById('uploadForm');
        const fileInput = document.getElementById('fileInput');
        const docStatusSpan = document.getElementById('doc-status');
        const uploadButton = document.getElementById('uploadButton');
        const queryForm = document.getElementById('queryForm');
        const questionInput = document.getElementById('questionInput');
        const askButton = document.getElementById('askButton');
        const chatMessages = document.getElementById('chat-messages');

        function showToast(message, type = 'success') {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = 'toast show ' + type;
            setTimeout(() => { toast.className = 'toast'; }, 3000);
        }

        function addMessage(content, type) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            const avatar = `<div class="avatar">${type === 'user' ? 'You' : 'AI'}</div>`;
            const messageContent = `<div class="content">${content}</div>`;
            messageDiv.innerHTML = avatar + messageContent;
            chatMessages.prepend(messageDiv);
        }

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                docStatusSpan.textContent = `Selected: ${fileInput.files[0].name}`;
            }
        });

        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (fileInput.files.length === 0) {
                showToast('Please select a file.', 'error');
                return;
            }
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            uploadButton.disabled = true;
            uploadButton.innerHTML = '<div class="spinner"></div>';
            docStatusSpan.textContent = 'Processing...';
            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                if (!response.ok) throw new Error((await response.json()).detail);
                const result = await response.json();
                showToast(result.message, 'success');
                questionInput.disabled = false;
                askButton.disabled = false;
                docStatusSpan.textContent = `Loaded: ${fileInput.files[0].name}`;
                addMessage(`Document "${fileInput.files[0].name}" is ready. You can now ask questions.`, 'ai');
            } catch (error) {
                showToast(error.message, 'error');
                docStatusSpan.textContent = 'Upload failed.';
            } finally {
                uploadButton.disabled = false;
                uploadButton.textContent = 'Process Document';
            }
        });

        queryForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const question = questionInput.value.trim();
            if (!question) return;
            addMessage(question, 'user');
            questionInput.value = '';
            askButton.disabled = true;
            addMessage('<div class="spinner"></div>', 'ai');
            const thinkingMessage = chatMessages.firstChild;
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: new URLSearchParams({ question })
                });
                if (!response.ok) throw new Error((await response.json()).detail);
                const result = await response.json();
                thinkingMessage.querySelector('.content').innerHTML = result.answer;
            } catch (error) {
                thinkingMessage.querySelector('.content').innerHTML = `Error: ${error.message}`;
            } finally {
                askButton.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

# --- API Endpoints ---

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        # Run the synchronous, CPU-bound function in a separate thread
        num_chunks = await asyncio.to_thread(process_and_store_document_sync, file.filename, file_content)
        return {"message": f"Successfully processed {file.filename} ({num_chunks} chunks stored)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_ai(question: str = Form(...)):
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        # Run the synchronous, CPU-bound function in a separate thread
        answer = await asyncio.to_thread(query_document_sync, question)
        return {"answer": answer}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    return html_content

# --- Main entry point ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
