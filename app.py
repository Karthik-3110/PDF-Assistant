import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Streamlit Page Setup
st.set_page_config(page_title="Academic PDF Assistant", layout="centered")
st.title("📘 Academic PDF Assistant")
st.caption("FAISS + Groq (LLaMA-3.1 Cloud)")


# Session State (Chat History)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])


@st.cache_resource(show_spinner=False)
def process_pdf(file_bytes):
    """Load PDF, split text, embed and index with FAISS"""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        pdf_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    page_count = len(docs)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, page_count


# Main App Logic
if uploaded_file:
    with st.spinner("Processing PDF..."):
        vectorstore, page_count = process_pdf(uploaded_file.read())

    st.success(f"PDF ready | Pages: {page_count} 📄")


    # Groq LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=300
    )


    # Prompt (Hybrid RAG + Reasoning)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an academic assistant.

First, try to answer using the provided document context.
If the context is insufficient, use your own knowledge to give a correct and helpful answer.
Keep the response concise and academic.

Document Context:
{context}

Question:
{question}

Answer:
"""
    )


    # Display Chat History
    st.subheader("💬 Chat")

    for role, message in st.session_state.chat_history:
        if role == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(message)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message)


    # Chat Input (Auto-clear)
    user_question = st.chat_input("Ask a question")

    if user_question:
        with st.spinner("Thinking..."):
            docs = vectorstore.similarity_search(user_question, k=3)
            context = "\n\n".join(d.page_content[:500] for d in docs)

            final_prompt = prompt.format(
                context=context,
                question=user_question
            )

            response = llm.invoke(final_prompt)

        # Save conversation
        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("assistant", response.content))

        st.rerun()

else:
    st.info("Upload a PDF to start chatting")
