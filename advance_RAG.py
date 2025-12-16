from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader, WebBaseLoader
)

from dotenv import load_dotenv

# Load environment variables from .env file
# Make sure your .env file contains:  OPENAI_API_KEY="your_api_key"
load_dotenv()

docs = []
docs.extend(TextLoader("data/sample.txt", encoding="utf-8").load())
docs.extend(PyPDFLoader("data/sample.pdf").load())
docs.extend(Docx2txtLoader("data/sample.docx").load())
docs.extend(WebBaseLoader(["https://example.com"]).load())

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(chunks, embeddings)

from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(chunks)
bm25.k = 6

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

def hybrid_retrieve(query: str):
    docs = bm25.invoke(query) + vector_retriever.invoke(query)

    seen = set()
    unique_docs = []
    for d in docs:
        if d.page_content not in seen:
            unique_docs.append(d)
            seen.add(d.page_content)

    return unique_docs

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the query to be explicit and retrieval-friendly.

Query: {query}
Rewritten:
""")

def rephrase_query(query: str) -> str:
    return (rewrite_prompt | llm).invoke({"query": query}).content.strip()

from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, docs, top_k=4):
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [doc for doc, _ in ranked[:top_k]]

answer_prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
Cite sources using [source_name].

Context:
{context}

Question:
{question}

Answer with citations:
""")

def advanced_rag(question: str):
    # 1. Rephrase
    refined_query = rephrase_query(question)

    # 2. Hybrid retrieval
    retrieved_docs = hybrid_retrieve(refined_query)

    # 3. Cross-encoder reranking
    top_docs = rerank(refined_query, retrieved_docs)

    # 4. Build cited context
    context_blocks = []
    for i, doc in enumerate(top_docs):
        source = doc.metadata.get("source", f"doc_{i}")
        context_blocks.append(f"[{source}]\n{doc.page_content}")

    context = "\n\n".join(context_blocks)

    # 5. Synthesize answer
    response = (answer_prompt | llm).invoke({
        "context": context,
        "question": question
    })

    return response.content, top_docs

answer, sources = advanced_rag("What is compliance?")

print("ANSWER:\n", answer)
print("\nCITED SOURCES:")
for s in sources:
    print("-", s.metadata.get("source"))
