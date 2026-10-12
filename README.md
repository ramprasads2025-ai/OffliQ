# OffliQ

*An Offline AI Study Companion*

OffliQ lets students upload lecture notes and academic PDFs, ask questions about them, and get clear explanations from an AI model that runs **entirely on their own device**. No internet connection is needed once it is set up.

## Problem Statement

Students often rely on internet connectivity and cloud-based AI tools to understand lecture notes and academic documents. Unreliable connectivity and limited data can make accessing these tools difficult.

OffliQ helps students upload academic documents, ask questions about their content, and receive understandable explanations through offline document processing and locally available AI capabilities.

## Objectives

- Enable students to access saved study materials offline.
- Extract and search text from uploaded PDF documents.
- Answer questions using relevant document content.
- Simplify complex academic concepts.
- Help students prepare for examinations.

## Features

- **PDF upload and offline storage:** documents are saved in a local SQLite database and stay available without internet.
- **Document-based question answering:** answers come only from the student's own document, with page-numbered sources.
- **Simplified and exam-oriented explanations:** choose Standard, "Simplify (like I'm 15)", or Exam-oriented answer styles.
- **Chapter summaries:** pick a page range and get an overview, key points and key terms.
- **Practice question generation:** create multiple-choice or short-answer questions from any page range.
- **Offline indicator:** the sidebar shows whether the app is running offline and whether the local AI is ready.

## How It Works

1. **Extract:** text is pulled from each PDF page with PyMuPDF.
2. **Index:** the text is split into overlapping chunks and converted to embeddings by a local model. If the embedding model is unavailable, the app falls back to keyword search.
3. **Retrieve:** a question is matched against the stored chunks to find the most relevant passages.
4. **Answer:** a local language model writes the answer using only those passages, and the sources are shown.

Everything runs on the user's device through [Ollama](https://ollama.com). No document or question ever leaves the computer.

## Technology Stack

| Part | Technology |
|---|---|
| Interface | Streamlit (runs locally in the browser) |
| Local language model | Ollama with `llama3.2:3b` |
| Embeddings | Ollama with `nomic-embed-text` |
| PDF text extraction | PyMuPDF |
| Storage | SQLite |
| Search | NumPy (cosine similarity) with keyword fallback |
| Language | Python 3.12 |

## Setup

You need an internet connection **once**, to download the tools and models. After that, OffliQ works fully offline.

1. **Install Ollama** from [ollama.com](https://ollama.com) and make sure it is running.
2. **Download the models** in a terminal:
   ```
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
3. **Install Python 3.12** from [python.org](https://www.python.org/downloads/). Very new Python versions (3.15) may fail to install the dependencies.
4. **Download this project** (green *Code* button, then *Download ZIP*) and unzip it.
5. **Install the dependencies** from inside the project folder:
   - Mac/Linux: `python3.12 -m pip install -r requirements.txt`
   - Windows: `python -m pip install -r requirements.txt`
6. **Run the app:**
   - Mac/Linux: `python3.12 -m streamlit run app.py`
   - Windows: `python -m streamlit run app.py`

The app opens in your browser at `http://localhost:8501`.

## How to Use

1. Open the **Library** tab, upload a PDF and click **Save to library**.
2. Open **Ask**, choose a document and an answer style, and type a question.
3. Use **Chapter summary** and **Practice questions** to revise a page range.

To test offline mode, turn off your wifi and keep using the app.

## Limitations

- Scanned PDFs (images of pages) are not supported because OCR is not included yet.
- Summaries and practice questions use up to about 2,500 words at a time, so smaller page ranges give better results.
- Answer quality depends on the size of the local model.

## Project Status

Working prototype built during a hackathon.

## Future Scope

- OCR support for scanned notes
- Support for additional document formats (DOCX, PPTX, text)
- Multilingual explanations
- Flashcards and saved chat history
- Personalized learning recommendations
- A packaged desktop app with one-click install

## Team Members

- Ram Prasad S.
- Nivedita B J.
- Rohini Sanjaya.
- Netradarsini S.
- Nikhita S Nair.

## License

MIT License
