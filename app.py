import asyncio
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Ensure asyncio works in Streamlit
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ---- Configure Gemini API ----
def configure_gemini():
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    return api_key

# ---- Utility Functions ----
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create in-memory Chroma (no SQLite)
    vectorstore = Chroma.from_texts(
        chunks,
        embedding=embeddings,
        persist_directory=None,  # disables SQLite persistence
        client_settings={"chroma_db_impl": "in_memory"}
    )

    st.session_state.vectorstore_created = True
    st.session_state.vectorstore = vectorstore  # store in session

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say "answer is not available in the context".
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question"}
    ]

def user_input(user_question):
    if not st.session_state.get("vectorstore_created", False):
        st.warning("Please upload and process PDFs first.")
        return {"output_text": ["No vectorstore found."]}

    vectorstore = st.session_state.vectorstore
    docs = vectorstore.similarity_search(user_question, k=3)
    chain = get_conversational_chain()
    return chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

# ---- Streamlit App ----
def main():
    configure_gemini()
    st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")
    st.title("Chat with PDF files 🤖")

    # Sidebar for PDF upload & processing
    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader("Select PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ PDFs processed successfully!")
            else:
                st.warning("Please select PDF files first.")

        st.header("Chat Options")
        st.button("Clear Chat History", on_click=clear_chat_history)

    if "messages" not in st.session_state:
        clear_chat_history()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                output_text = response.get("output_text", ["No answer available"])
                full_response = ""
                placeholder = st.empty()
                for item in output_text:
                    full_response += item
                    placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
