# Codex Historica 📜

An industry-standard AI research assistant that allows you to upload and chat with historical documents. This project implements a sophisticated **Retrieval-Augmented Generation (RAG)** pipeline, leveraging a state-of-the-art language model to provide accurate, context-aware answers grounded in the provided text.

This application is more than just a Q&A bot; it's a demonstration of an end-to-end AI system capable of handling unstructured data (images and PDFs) and transforming it into a searchable, intelligent knowledge base.

![Codex Historica Screenshot](screenshot.png)

---

## ✨ Core Features

-   **Multi-Format Document Ingestion:** Seamlessly upload and process both single-page images (PNG, JPG) and complex, multi-page PDF documents.
-   **High-Fidelity OCR Engine:** Utilizes the powerful Tesseract engine to automatically extract text from documents. For PDFs, each page is rendered at a high resolution (300 DPI) to ensure maximum accuracy, even with varied fonts and layouts.
-   **State-of-the-Art Language Model:** Powered by **Google's `FLAN-T5-Large`**, a highly capable instruction-tuned model, to provide nuanced, coherent, and contextually accurate answers.
-   **Advanced k-NN Vector Search:** Implements a k-Nearest Neighbors search using a `sentence-transformers` model to embed text chunks and a `ChromaDB` vector store. This allows the system to instantly find the most semantically relevant information to answer a user's query.
-   **Professional & Reactive UI:** The user interface is built from the ground up to be modern, intuitive, and fully responsive. It features a "Glassmorphism" design, a live chat window, toast notifications for status updates, and loading indicators for a smooth user experience.
-   **Robust Asynchronous Backend:** Built with **FastAPI**, the backend is fully asynchronous. This allows it to handle long-running, CPU-bound tasks (like OCR and model inference) in separate threads, preventing the server from freezing and ensuring the UI remains responsive at all times.

---

## 🛠️ Technology Choices & Rationale

This project uses a carefully selected stack of modern tools chosen for their performance, scalability, and industry relevance.

-   **Backend:** **Python** with **FastAPI** and **Uvicorn**.
    -   *Why?* FastAPI's asynchronous nature is perfect for I/O-bound operations and its modern type-hinting system leads to robust, easy-to-read code. Uvicorn is the industry-standard ASGI server for running it.

-   **AI Pipeline:**
    -   **Large Language Model (LLM):** **`google/flan-t5-large`** from Hugging Face `transformers`.
        -   *Why?* FLAN-T5 is instruction-tuned, making it exceptionally good at following the complex "answer only from context" prompts required for a RAG system. The `large` version provides a strong balance of performance and high-quality generation.
    -   **Embeddings (k-NN Search):** **`all-MiniLM-L6-v2`** from `sentence-transformers`.
        -   *Why?* This model is a top-performer for generating sentence embeddings. It's lightweight, fast, and highly effective at capturing semantic meaning, making it ideal for the "retrieval" part of the RAG pipeline.
    -   **Vector Database:** **`ChromaDB`**.
        -   *Why?* ChromaDB is an open-source, developer-friendly vector store that is easy to set up and run locally without needing a separate database server, making it perfect for a self-contained project.
    -   **OCR Engine:** **`Tesseract`** via `pytesseract`.
        -   *Why?* Tesseract is a powerful, open-source OCR engine capable of recognizing a wide variety of fonts and languages.
    -   **PDF Processing:** **`PyMuPDF`**.
        -   *Why?* It is a highly performant and versatile library for accessing PDF content, allowing for high-DPI rendering of pages, which is critical for OCR accuracy.

-   **Core Libraries:** **PyTorch**, **Pillow**.
    -   *Why?* PyTorch is the underlying tensor library for the Hugging Face models. Pillow is the essential library for image manipulation in Python.

---

## 🏛️ Detailed Project Architecture (The RAG Pipeline)

This project is a practical implementation of the **Retrieval-Augmented Generation (RAG)** architecture. This design enhances the capabilities of Large Language Models by grounding them in specific, user-provided information.

Here is a step-by-step breakdown of the data flow:

1.  **Ingestion & Preprocessing:**
    -   A user uploads a document (Image or PDF) via the web interface.
    -   The FastAPI backend receives the file. If it's a PDF, `PyMuPDF` iterates through each page, rendering it as a high-resolution (300 DPI) image.
    -   Each image is passed to the `Tesseract` OCR engine, which extracts the raw text.

2.  **Indexing (Creating the Knowledge Base):**
    -   The extracted text is segmented into smaller, overlapping chunks (paragraphs).
    -   The `sentence-transformers` embedding model (`all-MiniLM-L6-v2`) converts each text chunk into a high-dimensional vector. This vector numerically represents the semantic meaning of the chunk.
    -   These vectors, along with their corresponding text, are stored in a `ChromaDB` vector database. This indexed database now serves as the document's "memory."

3.  **Retrieval:**
    -   When a user asks a question in the chat, the same embedding model converts the question into a vector.
    -   The system performs a **k-Nearest Neighbors (k-NN)** similarity search in ChromaDB, comparing the question's vector against all the stored text chunk vectors.
    -   The top `k` (in our case, 5) most similar text chunks are retrieved. These chunks are the most relevant pieces of context from the document for answering the user's question.

4.  **Generation:**
    -   A carefully engineered prompt is constructed. This prompt contains strict instructions, the retrieved context chunks, and the user's original question.
    -   This complete prompt is passed to the **FLAN-T5-Large** model.
    -   The LLM, following the instructions, synthesizes the information *only* from the provided context to generate a final, human-readable answer.

5.  **Response:**
    -   The generated answer is sent back to the frontend and displayed in the chat window.

This entire process ensures that the AI's answers are not hallucinations but are directly tied to the content of the uploaded document.

---

## 🚀 How to Run Locally

Follow these steps to get the project running on your local machine.

**Prerequisites:**
-   Python 3.9+
-   Tesseract OCR Engine ([Installation Guide for Windows](https://github.com/UB-Mannheim/tesseract/wiki))
-   Microsoft C++ Build Tools ([Download Here](https://visualstudio.microsoft.com/visual-studio-build-tools/))

**Installation:**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/pavangunnam19/Codex-Historica.git](https://github.com/pavangunnam19/Codex-Historica.git)
    cd Codex-Historica
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows, it is highly recommended to use the x64 Native Tools Command Prompt for VS 2022
    venv\Scripts\activate
    ```

3.  **Install dependencies from the requirements file:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    uvicorn main:app --reload
    ```
    -   **Note:** The first time you run this, it will take several minutes to download the ~3GB FLAN-T5-Large model. This is a one-time download.

5.  Open your browser and navigate to `http://127.0.0.1:8000`.

---

## 🧠 Challenges & Learnings

-   **Dependency Management on Windows:** A significant challenge was installing packages with C++ extensions (like `sentencepiece`) on Windows. This was solved by using the official Microsoft C++ Build Tools and the specific "x64 Native Tools Command Prompt" to ensure the compiler was available in the system's PATH.
-   **OCR Quality:** Initial tests with PDFs yielded poor quality text. This was resolved by increasing the rendering resolution to 300 DPI in the `PyMuPDF` processing step, dramatically improving the input quality for Tesseract.
-   **Prompt Engineering:** Early versions of the AI would sometimes answer from general knowledge instead of the document. The solution was to engineer a much stricter prompt, explicitly instructing the model to answer *only* from the provided context, which significantly improved the faithfulness of the responses.
-   **Asynchronous Task Handling:** The initial synchronous implementation would freeze when processing large documents. This was rectified by refactoring the AI logic into synchronous functions and using FastAPI's `asyncio.to_thread` to run them in a separate thread, keeping the server responsive.

---
