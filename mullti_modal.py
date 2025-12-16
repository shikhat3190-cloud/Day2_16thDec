from langchain_community.document_loaders import TextLoader

from dotenv import load_dotenv

# Load environment variables from .env file
# Make sure your .env file contains:  OPENAI_API_KEY="your_api_key"
load_dotenv()

text_docs = TextLoader(
    "sample.txt",
    encoding="utf-8"
).load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

text_chunks = splitter.split_documents(text_docs)

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()

text_vectorstore = FAISS.from_documents(
    text_chunks,
    embeddings
)

from langchain_core.documents import Document

import base64

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


image_docs = [
    Document(
        page_content="",
        metadata={
            "type": "image",
            "path": "RAG.jpeg",
            "description": "System architecture diagram"
        }
    ),
    Document(
        page_content="",
        metadata={
            "type": "image",
            "path": "JAVA.jpeg",
            "description": "Workflow diagram"
        }
    )
]

def retrieve_multimodal(query: str, k=4):
    text_results = text_vectorstore.similarity_search(query, k=k)

    relevant_images = image_docs

    return text_results, relevant_images

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

def multimodal_answer(question: str):
    text_docs, image_docs = retrieve_multimodal(question)

    # Prepare text context
    text_context = "\n\n".join(
        doc.page_content for doc in text_docs
    )

    # Prepare image inputs
    image_messages = []

    for img in image_docs:
        image_b64 = image_to_base64(img.metadata["path"])

        image_messages.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"
            }
        })

    # Build multimodal message
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"""
            Answer the question using the text and images provided.
            
            Text Context:
            {text_context}
            
            Question:
            {question}
            """
            },
            *image_messages
        ]
    )

    response = llm.invoke([message])
    return response.content

answer = multimodal_answer(
    "Explain the system architecture of JAVA"
)

print(answer)



