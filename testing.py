# ******************************************** Imports ********************************************
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import uuid
from chat_bot_backend import graph_chatbot
from langchain_community.document_loaders import PyPDFLoader
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
import tempfile

# ******************************************** Utility Functions ********************************************
def get_new_thread_id():
    return str(uuid.uuid4())

def store_thread_id(thread_id):
    if thread_id not in st.session_state['stored_thread_ids']:
        st.session_state['stored_thread_ids'].append(thread_id)

def new_chat_button():
    st.session_state['thread_id'] = get_new_thread_id()
    store_thread_id(st.session_state['thread_id'])
    st.session_state["chat_titles"][st.session_state['thread_id']] = "New chat..."
    st.session_state['messages'] = []

def display_chat(thread_id):
    state = graph_chatbot.get_state(config={'configurable': {'thread_id': thread_id}}) or {}
    return state.get('values', {}).get('messages', [])

# ******************************************** RAG Utilities ********************************************
def extract_text(file_path):
    """Extract text from PDF, DOCX, or TXT."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return "\n".join([p.page_content for p in pages])
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return None

def create_vectorstore(text):
    """Create an in-memory Chroma vectorstore from document text."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Use a temporary directory to avoid persistent conflicts
    persist_dir = tempfile.mkdtemp(prefix="chroma_store_")
    vectordb = Chroma.from_texts(chunks, embeddings, collection_name="uploaded_docs", persist_directory=persist_dir)
    return vectordb

def retrieve_context(query, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([d.page_content for d in docs])

# ******************************************** Session State Initialization ********************************************
if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}
if 'stored_thread_ids' not in st.session_state:
    st.session_state['stored_thread_ids'] = []
if "messages" not in st.session_state:
    st.session_state['messages'] = []
if "thread_id" not in st.session_state:
    st.session_state['thread_id'] = get_new_thread_id()
if "vectordb" not in st.session_state:
    st.session_state["vectordb"] = None

# ******************************************** Sidebar ********************************************
st.sidebar.title("LLM Chatbot")

# 🔹 File upload for RAG
uploaded_file = st.sidebar.file_uploader("Upload a document (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
    text = extract_text(file_path)
    if text:
        st.session_state["vectordb"] = create_vectorstore(text)
        st.sidebar.info("✅ File indexed for RAG search!")
    else:
        st.sidebar.error("⚠️ Could not extract text from this file.")

# Sidebar chat management
if st.sidebar.button("New chat"):
    new_chat_button()

st.sidebar.header("My conversations:")
for id in st.session_state["stored_thread_ids"]:
    title = st.session_state["chat_titles"].get(id, id[:5] + "...")
    if st.sidebar.button(title, key=id):
        st.session_state['thread_id'] = id
        messages = display_chat(id)

        temp_msg = []
        for i in messages:
            if isinstance(i, HumanMessage):
                role, avatar = "user", "🧑"
            elif isinstance(i, AIMessage):
                role, avatar = "assistant", "🤖"
            else:
                role, avatar = "system", "⚙️"
            temp_msg.append({"role": role, "content": i.content, "avatar": avatar})
        st.session_state['messages'] = temp_msg

# ******************************************** Display Chat ********************************************
for msg in st.session_state['messages']:
    with st.chat_message(msg["role"], avatar=msg["avatar"]):
        st.text(msg["content"])

store_thread_id(st.session_state['thread_id'])
config = {'configurable': {'thread_id': st.session_state['thread_id']}}

# ******************************************** Chat Input ********************************************
user_input = st.chat_input("Enter your message:")

if user_input:
    # Display user input immediately
    st.session_state['messages'].append({"role": "user", "content": user_input, "avatar": "🧑"})
    with st.chat_message("user", avatar="🧑"):
        st.text(user_input)

    # 🔹 Retrieve augmented context if file uploaded
    if st.session_state["vectordb"]:
        context = retrieve_context(user_input, st.session_state["vectordb"])
        augmented_query = f"Context:\n{context}\n\nQuestion: {user_input}"
        user_msg = HumanMessage(content=augmented_query)
    else:
        user_msg = HumanMessage(content=user_input)

    # 🔹 Send to chatbot backend
    # 🔹 Send to chatbot backend (LangGraph expects a dict input)
    try:
        response = graph_chatbot.invoke({"messages": [user_msg]}, config=config)

        # The graph likely returns a dict containing 'messages'
        if isinstance(response, dict) and "messages" in response:
            last_msg = response["messages"][-1]
            ai_response = last_msg.content if isinstance(last_msg, AIMessage) else str(last_msg)
        else:
            ai_response = str(response)

    except Exception as e:
        ai_response = f"⚠️ Error generating response: {e}"


    # 🔹 Display and store assistant message
    st.session_state['messages'].append({"role": "assistant", "content": ai_response, "avatar": "🤖"})
    with st.chat_message("assistant", avatar="🤖"):
        st.text(ai_response)
