# Conversational RAG PDF QA with Chat History

A **Conversational RAG (Retrieval-Augmented Generation)** system built using **Python**, **Streamlit**, **Groq API (Gemma 2-9b-it)**, and **LangChain**.

It allows you to upload PDFs and interactively ask questions about their content, with full chat history context.

-----

## 🚀 Features

  * 📄 **Conversational PDF QA** – Upload one or multiple PDFs and ask follow-up questions.
  * 🔍 **RAG Pipeline** – Retrieval-augmented QA for more accurate answers.
  * 🧠 **History-Aware Retrieval** – Maintains conversation context using session-based chat memory.
  * 🔑 **Session Management** – Supports multiple sessions via `session_id`.
  * ⚡ **Fast & Simple Frontend** – Built with Streamlit for rapid interaction.

-----

## 🧰 Tech Stack

| Component      | Technology                                    |
| :------------- | :-------------------------------------------- |
| **Language** | Python                                        |
| **Frontend** | Streamlit                                     |
| **LLM Backend**| Groq (`gemma2-9b-it`)                         |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace)              |
| **Vector Store**| Chroma                                        |
| **PDF Loader** | LangChain `PyPDFLoader`                       |
| **RAG Tools** | LangChain's `create_retrieval_chain`, `RunnableWithMessageHistory` |

-----

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ayush-Rawat-1/Groq_QA_pdf.git
cd conversational-pdf-rag
```

### 2️⃣ Create a Virtual Environment

```bash
conda create -n rag_pdf python=3.12
conda activate rag_pdf
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

-----

## 🔧 Environment Setup

Create a `.env` file and add your **Hugging Face token**:

```
HF_TOKEN=your_huggingface_token_here
```

At runtime, you will input your **Groq API key** in the Streamlit UI.

-----

## 📝 Usage

Run the Streamlit app:

```bash
streamlit run Groq_QA_pdf.py
```

### Workflow:

1.  Upload PDF(s)
2.  Enter your **Groq API Key**
3.  Provide a **Session ID** (default is `default_session`)
4.  Ask your questions\!
5.  All chat history is stored and used for follow-up queries.

-----

## 🧠 How It Works

  * **Document Processing:**
    PDFs are split into chunks with `RecursiveCharacterTextSplitter`, embedded using `all-MiniLM-L6-v2`, and stored in **Chroma**.

  * **Contextualized QA:**
    Uses `create_history_aware_retriever` to reformulate questions when needed, maintaining chat context.

  * **LLM Response:**
    Queries are answered by `gemma2-9b-it` via **Groq API**, using relevant document context.

  * **Session Memory:**
    Chat history is stored per session using `ChatMessageHistory` in Streamlit's `session_state`.

-----

## 🗂️ Project Structure

```
Groq_QA_pdf.py        # Main Streamlit Application
.env                  # Environment Variables (HF_TOKEN)
requirements.txt      # Python Dependencies
```

-----

## 🛠️ Requirements

  * Python 3.10+
  * Groq API Key
  * Hugging Face Account & Token

-----

## ⚠️ Notes

  * **Model:**
    Uses **Gemma 2-9b-it** by default. You can modify the code to try other models via Groq.

-----

## 💡 Future Enhancements

  * Add support for DOCX/TXT file uploads.
  * Include local model options for offline usage.
  * Export and visualize chat history.

-----

## 📄 License

This project is open-source and intended for educational and personal use.