


import os

from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# =====================================================
# 1. SET ENVIRONMENT VARIABLES
# =====================================================
# Gemini API Key from Google AI Studio
os.environ["GOOGLE_API_KEY"] = "AIzaSyBmxzbCe2KclpmXTRV5fwAL1POmkhSAYuI"

# Zilliz Cloud credentials
ZILLIZ_ENDPOINT = "https://in03-df0c9fc1ceb58be.serverless.aws-eu-central-1.cloud.zilliz.com"
ZILLIZ_API_KEY = "f58f074943e1058022593921b0d058ea93ece8af8cfbb60fb5c1d4ca96ed027ce3020a43b34119e6a1e8ecea56f926bbbb390278"

# =====================================================
# 2. SAMPLE KNOWLEDGE BASE (EMBEDDED DATA)
# =====================================================
texts = [
    "LangChain is a framework for building applications using large language models.",
    "Milvus is a vector database designed for fast similarity search on embeddings.",
    "Zilliz Cloud is a managed cloud service built on top of Milvus.",
    "RAG retrieves relevant documents from a vector database before generating an answer using an LLM."
]

# =====================================================
# 3. EMBEDDINGS (LOCAL, FREE)
# =====================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =====================================================
# 4. VECTOR DATABASE (ZILLIZ / MILVUS CLOUD)
# =====================================================
vector_db = Milvus.from_texts(
    texts=texts,
    embedding=embeddings,
    collection_name="rag_demo_collection",
    connection_args={
        "uri": ZILLIZ_ENDPOINT,
        "token": ZILLIZ_API_KEY
    },
    drop_old=True
)

# =====================================================
# 5. RETRIEVER WITH RELEVANCE THRESHOLD
# =====================================================
retriever = vector_db.as_retriever(
    search_kwargs={
        "k": 3,
        "score_threshold": 0.6
    }
)

# =====================================================
# 6. LLM (GEMINI 2.5 FLASH)
# =====================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# =====================================================
# 7. RETRIEVAL QA CHAIN
# =====================================================
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# =====================================================
# 8. SUGGESTED QUESTIONS (GUIDED UX)
# =====================================================
SUGGESTED_QUESTIONS = [
    "What is LangChain?",
    "What is RAG?",
    "Why is Milvus used as a vector database?",
    "What is Zilliz Cloud?",
    "How does retrieval work in RAG?"
]

# =====================================================
# 9. INTERACTIVE QUERY LOOP
# =====================================================
print("\nRAG system is ready.")
print("You can ask questions related to the embedded knowledge base.")
print("Type 'exit' to quit.\n")

print("Suggested questions:")
for q in SUGGESTED_QUESTIONS:
    print(" -", q)
print()

while True:
    query = input("Your question: ").strip()

    if query.lower() == "exit":
        print("Exiting RAG system.")
        break

    # Use modern retriever API
    docs = retriever.invoke(query)

    # STRICT OUT-OF-SCOPE CHECK
    if not docs:
        print(
            "\nI don’t have enough information in my knowledge base to answer that.\n"
        )
        print("Try asking one of these questions:")
        for q in SUGGESTED_QUESTIONS:
            print(" -", q)
        print()
        continue

    # Call Gemini ONLY if relevant docs exist
    result = qa.invoke({"query": query})
    print("\nAnswer:", result["result"], "\n")
