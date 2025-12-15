# RAG Pipeline (Zilliz + Gemini)

## What this project does

This project demonstrates a **basic RAG (Retrieval Augmented Generation) system**.

In simple words:

* We store some text data as **vector embeddings**
* These embeddings are stored in an **online vector database (Zilliz / Milvus Cloud)**
* When a user asks a question:

  * Relevant data is retrieved from the vector DB
  * **Gemini 2.5 Flash** generates an answer **only from that data**
* If the question is outside the data, the system **politely refuses to answer**

---

## Tech stack used

* **Python**
* **LangChain** – orchestration
* **Zilliz Cloud (Milvus)** – online vector database
* **HuggingFace sentence-transformers** – local embeddings (free, no quota)
* **Gemini 2.5 Flash** – answer generation (LLM)

---

## What we implemented

* Created embeddings from sample text data
* Stored embeddings in **Zilliz Cloud**
* Built a retrieval-based QA pipeline using **LangChain**
* Added **dynamic user input** (CLI-based)
* Added **out-of-scope handling**:

  * If data is not present → system says it doesn’t know
* Added **suggested questions** to guide users

---

## How to run


    # activate virtual environment
    source venv/bin/activate
    
    # run the app
    python app.py


---

## Example behavior

### 1. In-scope question


    Your question: What is RAG?

    Answer:
    RAG (Retrieval Augmented Generation) retrieves relevant documents from a vector database
    before generating an answer using an LLM.


### 2. Out-of-scope question


    Your question: Who is the PM of India?

    Answer:
    I don't know the answer based on the provided context.


This shows the system **does not hallucinate** and stays within its knowledge base.


## Screenshots

Below are screenshots showing the working system:

* RAG system startup with suggested questions
* Correct answer for in-scope question (RAG)
* Safe refusal for out-of-scope question (PM of India)


<img width="466" height="152" alt="Screenshot 2025-12-15 at 7 14 39 PM" src="https://github.com/user-attachments/assets/54a4cb41-9d33-4f3e-a4e8-931135440fa5" />

<img width="1063" height="74" alt="Screenshot 2025-12-15 at 7 14 48 PM" src="https://github.com/user-attachments/assets/16209387-9a87-4114-8e38-739d9697393c" />

<img width="540" height="60" alt="Screenshot 2025-12-15 at 7 14 53 PM" src="https://github.com/user-attachments/assets/49fd5d48-52a0-4e82-8df1-c55516d1d8ad" />


