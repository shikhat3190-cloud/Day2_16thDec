
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    WebBaseLoader
)

from dotenv import load_dotenv

# Load environment variables from .env file
# Make sure your .env file contains:  OPENAI_API_KEY="your_api_key"
load_dotenv()

docs = []

# TXT
docs.extend(TextLoader("data/sample.txt", encoding="utf-8").load())

# PDF
docs.extend(PyPDFLoader("data/sample.pdf").load())

# DOCX
docs.extend(Docx2txtLoader("data/sample.docx").load())

# Web page
docs.extend(
    WebBaseLoader(["https://en.wikipedia.org/wiki/Regulatory_compliance"]).load()
)

print(f"Loaded {len(docs)} raw documents")

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

#### ------- Create vectorstore------------
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

#### ------- Create Retriever ------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

### --------Add LLM + Response Synthesis (AG in RAG) ----------
# You are an assistant answering questions using only the provided context.
prompt = ChatPromptTemplate.from_template("""

Context:
{context}

Question:
{question}

Answer:
""")

### --------Add LLM + Response Synthesis (AG in RAG) ----------
# You are an assistant answering questions using only the provided context.
prompt_constraint = ChatPromptTemplate.from_template("""
You are a helpful assistant who only uses the given context below to answer questions. IF the answer is not present in the
context politely decline user request.

Context:
{context}

Question:
{question}

Answer:
""")

def rag_answer(question: str,constraint: bool):
    # Retrieve
    retrieved_docs = retriever.invoke(question)

    # Combine context
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Synthesize response
    # LCEL - Langchain expression language
    if not constraint:
        chain = prompt | llm
    else:
        chain = prompt_constraint | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content, retrieved_docs

constraint = True
question = ["What is company leave policy","who is the president of USA?"]

for q in question:
    answer, sources = rag_answer(q,constraint)
    print("ANSWER:\n", answer)
    print("\nSOURCES:")
    for s in sources:
        print("-",s.page_content)
        print("-", s.metadata.get("source"))






