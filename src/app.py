import os
import base64
import gc
import random
import tempfile
import time
import uuid
from IPython.display import Markdown, display
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
import streamlit as st
from dotenv import load_dotenv



load_dotenv()  # Load .env file


if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

@st.cache_resource
def load_llm(selected_llm, temperature):
    if selected_llm=="Ollama3.2 3b":
        llm_name = "ollama3.2"
    elif selected_llm=="DeepSeek R1":
        llm_name = "deepseek-r1"
    
    llm = Ollama(model=llm_name, request_timeout=120.0, temperature=temperature)
    return llm

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

# def display_pdf(file):
#     st.markdown("### PDF Preview")
#     base64_pdf = base64.b64encode(file.read()).decode("utf-8")
#     pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
#                         style="height:100vh; width:100%"
#                     >
#                     </iframe>"""
#     st.markdown(pdf_display, unsafe_allow_html=True)

def display_pdf(file):
    st.markdown("### PDF Preview")
    try:
        # Read the file as bytes
        base64_pdf = base64.b64encode(file.read()).decode("utf-8")
        # Embed PDF in an iframe
        pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}" 
                width="100%" height="100%" 
                style="height:100vh; width:100%; border:none;">
        </iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")

# Sidebar Dropdown (Select Box)
selected_llm = st.sidebar.selectbox(
    "Choose a model:",
    ["Ollama3.2 3b", "DeepSeek R1"]  # Dropdown values
)

# Display the selected value
st.write(f"You selected: {selected_llm}")

st.sidebar.subheader("Adjust Temperature")
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05,
    help="Controls the randomness of the model's responses. Lower values make the output more deterministic.",
)

with st.sidebar:
    st.header(f"Browse document")
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing document...")

                if file_key not in st.session_state.get('file_cache', {}):
                    if os.path.exists(temp_dir):
                            loader = SimpleDirectoryReader(
                                input_dir = temp_dir,
                                required_exts=[".pdf"],
                                recursive=True
                            )
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    
                    docs = loader.load_data()
                    llm = load_llm(selected_llm, temperature)
                    embed_model = HuggingFaceEmbedding( model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
                    Settings.embed_model = embed_model
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)

                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True)

                    qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )
                    
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                st.success("Chat with your document!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     



col1, col2 = st.columns([12, 1], vertical_alignment="center")

with col1:
    # st.header(f"DocuChat")
    # st.header(f"An open-source conversational AI using Llama-3.3 70B")
    # st.markdown(
    #     "<h4 style='text-align: center; color: white;'>Developed by Amir Kamel</h4>",
    #     unsafe_allow_html=True,
    # )
    st.header("DocuChat :books:",divider="blue")
    st.subheader("An open-source conversational AI")






# with col2:
#     st.button("Clear ↺", on_click=reset_chat)

if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        streaming_response = query_engine.query(prompt)
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer - Positioned at the bottom of the page
footer_placeholder = st.empty()

footer_placeholder.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #0E1117;
            text-align: center;
            font-size: 11px;
            color: gray;
            padding: 10px;
            z-index: 9999;  /* Ensure it stays on top */
        }
    </style>
    <div class="footer">
        &copy; 2025 Amir Kamel. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True,
)

# Add space below the chat input box so footer stays visible
st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)