import os
import base64
import gc
import tempfile
import uuid
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

load_dotenv()

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id

@st.cache_resource
def load_llm(selected_llm, temperature):
    llm_mapping = {
        "llama3.2 3b": "llama3.2:latest",
        "DeepSeek R1": "deepseek-r1:latest",
        "Text2SQL": "hf.co/yasserrmd/Text2SQL-1.5B-gguf:F16",
    }
    llm_name = llm_mapping.get(selected_llm, "llama3.2:latest")
    return Ollama(model=llm_name, request_timeout=120.0, temperature=temperature)

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(file):
    try:
        base64_pdf = base64.b64encode(file.read()).decode("utf-8")
        pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}" 
                width="100%" height="100%" 
                style="height:100vh; width:100%; border:none;"></iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")

st.sidebar.subheader("Select Model and Parameters")
selected_llm = st.sidebar.selectbox("Choose a model:", ["Ollama3.2 3b", "DeepSeek R1", "Text2SQL"])
st.write(f"You selected: {selected_llm}")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

st.sidebar.subheader("Upload Document (Optional)")
uploaded_file = st.sidebar.file_uploader("Choose your `.pdf` file", type="pdf")

# If a file is uploaded
if uploaded_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        file_key = f"{session_id}-{uploaded_file.name}"
        st.write("Indexing document...")
        
        if file_key not in st.session_state.file_cache:
            loader = SimpleDirectoryReader(input_dir=temp_dir, required_exts=[".pdf"], recursive=True)
            docs = loader.load_data()
            llm = load_llm(selected_llm, temperature)
            embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
            Settings.embed_model = embed_model
            index = VectorStoreIndex.from_documents(docs, show_progress=True)
            Settings.llm = llm
            query_engine = index.as_query_engine(streaming=True)
            st.session_state.file_cache[file_key] = query_engine
        else:
            query_engine = st.session_state.file_cache[file_key]
        
        st.success("Chat with your document!")
        display_pdf(uploaded_file)
# If no file is uploaded
else:
    # st.warning("No document uploaded. You can still chat with the selected LLM.")
    llm = load_llm(selected_llm, temperature)
    query_engine = None

if (selected_llm == "Text2SQL"):
    st.header("Text to SQL :page_facing_up:", divider="blue")
else:
    st.header("DocuChat :books:", divider="blue")
st.subheader("A conversational AI by Amir Kamel")

if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        if query_engine:
            streaming_response = query_engine.query(prompt)
            for chunk in streaming_response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
        else:
            raw_response = llm.complete(prompt)
            response_str = str(raw_response)

            # Splitting the query and explanation
            split_response = response_str.split("; ", 1)  # Splitting at first occurrence of "; "

            sql_query = split_response[0] + ";"  # Adding back the semicolon
            explanation = split_response[1] if len(split_response) > 1 else "No explanation provided."

            # Display SQL Query
            st.markdown("### Generated SQL Query:")
            st.code(sql_query, language="sql")

            # Display Explanation
            st.markdown("### Explanation:")
            st.markdown(explanation)
            
            # message_placeholder.markdown(raw_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
