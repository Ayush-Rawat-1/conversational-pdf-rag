# Conversational RAG PDF QA with Chat History

A **Conversational RAG (Retrieval-Augmented Generation)** system built using **Python**, **Streamlit**, **Groq API (Gemma 2-9b-it)**, and **LangChain**.

It allows user to upload PDFs and interactively ask questions about their content, with full chat history context.

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
git clone https://github.com/Ayush-Rawat-1/conversational-pdf-rag
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


# **🚀 High-Level Overview**

The project implements a **Conversational Retrieval-Augmented Generation (RAG)** pipeline using:

* **Groq LLM (Gemma-2 9b IT model)** for answering questions
* **PDF document retrieval** for providing factual answers
* **Chat history management** for follow-up questions with context
* **Streamlit UI** for interaction

---

# **📂 Step-by-Step Code Flow**

## **1️⃣ Setup and Imports**

### Libraries used:

* `Streamlit` for frontend UI
* `LangChain` for building the RAG pipeline
* `Chroma` for vector database
* `HuggingFaceEmbeddings` for text embeddings
* `PyPDFLoader` for loading PDFs
* `dotenv` for environment management

---

## **2️⃣ Environment Setup**

```python
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
```

This loads your **Hugging Face token** from `.env` for embedding generation.

---

## **3️⃣ Streamlit Frontend**

```python
st.title("Conversational RAG with PDF and Chat History")
```

* Displays the app title and UI components for:

  * **Groq API Key input**
  * **PDF file upload**
  * **Session ID input**
  * **User question input**

---

## **4️⃣ Model and Vector Setup**

If the **Groq API key is provided**, it initializes:

```python
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

---

## **5️⃣ Upload PDF and Create Vector Store**

When PDFs are uploaded:

1. Each PDF is saved as a temporary file.
2. `PyPDFLoader` loads the text from PDFs into `document` objects.
3. Text is split into chunks using:

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
```

4. Each chunk is converted into embeddings via **HuggingFace** and stored in **Chroma** (vector database):

```python
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retrieval = vectorstore.as_retriever()
```

---

## **6️⃣ History-Aware Retriever**

A special prompt is created for **rephrasing follow-up questions** into **standalone questions**:

```python
contextualize_q_system_prompt = ChatPromptTemplate.from_messages([...])
```

Then:

```python
history_aware_retriever = create_history_aware_retriever(llm, retrieval, contextualize_q_system_prompt)
```

This makes the retriever context-aware by including past chat history.

---

## **7️⃣ Answer Generation**

Creates a **QA prompt** to instruct the LLM on how to answer:

```python
system_prompt = ("You are an assistant for question answering...")

qa_prompt = ChatPromptTemplate.from_messages([...])
```

Builds the QA pipeline:

```python
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
```

Combines **retriever + answer generator** into one:

```python
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

---

## **8️⃣ Chat History Management**

Session-based chat history is managed with:

```python
def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]
```

Each **`session_id`** stores a separate conversation history, allowing multiple users or sessions to run in parallel.

---

## **9️⃣ RunnableWithMessageHistory**

Wraps the RAG chain into a **conversational agent**:

```python
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chathistory",
    output_messages_key="answer"
)
```

This ensures the model sees **past messages** and stores new ones automatically.

---

## **🔄 User Interaction Loop**

```python
user_input = st.text_input("Your question:")
```

When the user asks a question:

1. Chat history for the session is retrieved.
2. The **conversational RAG chain** is invoked with:

```python
response = conversational_rag_chain.invoke(
    {"input": user_input},
    config={"configurable": {"session_id": session_id}}
)
```

3. The LLM:

   * Reformulates the question if needed.
   * Retrieves relevant PDF chunks.
   * Generates an answer based on the context.

4. The system updates chat history automatically.

5. The answer is displayed:

```python
st.success(f"Assistant: {response["answer"]}")
st.write("chat history:", session_history.messages)
```

---

# **📈 Flow Diagram (Textual)**

```
User uploads PDFs
         ↓  
PDFs processed into text chunks
         ↓  
Chunks embedded into vectors via HuggingFace → Stored in Chroma  
         ↓  
User asks a question 
         ↓  
Question reformulated if follow-up → History-Aware Retriever  
         ↓  
Relevant chunks retrieved from Chroma  
         ↓  
LLM (Groq Gemma2-9b-it) answers using the retrieved context 
         ↓  
Answer + Chat History updated → Displayed in Streamlit  
```

---

# **🧠 Why This Design?**

| Feature                     | Purpose                                |
| --------------------------- | -------------------------------------- |
| **LangChain RAG Chain**     | Combines retrieval + generation        |
| **History-Aware Retriever** | Handles follow-up questions properly   |
| **ChatMessageHistory**      | Manages multi-session conversations    |
| **Chroma Vector Store**     | Efficient document search              |
| **Streamlit UI**            | Easy interaction with PDFs and the bot |

---

# **✅ Summary**

This is a **modular and scalable RAG pipeline** for document Q\&A with conversation memory, ideal for:

* PDF Knowledge Bots
* AI Assistants for documents
* Research tools


## 🗂️ Project Structure

```
Groq_QA_pdf.py        # Main Streamlit Application
.env                  # Environment Variables (HF_TOKEN)
requirements.txt      # Python Dependencies
```

-----

## 🛠️ Requirements

  * Python 3.12+
  * Groq API Key
  * Hugging Face Account & Token

-----


## 📄 License

This project is open-source and intended for educational and personal use.