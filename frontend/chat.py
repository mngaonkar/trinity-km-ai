from math import log
from multiprocessing import Pipe
from venv import logger
import streamlit as st
from streamlit.components.v1 import html
from langchain_community.chat_models import ChatOllama
from backend.vectorstore import VectorStore
import constants
from langchain_core.language_models import LanguageModelInput
from backend.pipeline import Pipeline
from backend.loader import DocumentLoader
from backend.llm import LLM
import constants
from loguru import logger

class ChatGUI():
    """Chat GUI class for displaying chat messages."""
    def __init__(self, pipeline: Pipeline, vector_store: VectorStore):
        logger.info("Initializing chat GUI...")
        self.pipeline = pipeline
        self.pipeline.setup_session_state(st.session_state)
        pipeline.setup(vector_store)
        
        logger.info("Chat GUI initialized.")

    def set_vector_store(self, store: VectorStore):
        self.vector_store = store

    def model_changed(self):
        """Model changed."""
        pass

    def augmented_flag_changed(self):
        """Augmented flag changed."""
        self.pipeline.setup_pipeline()

    def run(self):
        # Set the page configuration
        st.set_page_config(
            page_title=constants.APP_NAME,
            page_icon=":speech_balloon:",
            layout="wide",
        )

        st.title(constants.APP_NAME + " " + constants.APP_VERSION)
        chat_container = st.container()
        sidebar_container = st.sidebar.container()

        # Save use specific settings in session state
        st.session_state["model"] = sidebar_container.selectbox("Model", constants.MODELS, on_change=self.model_changed)
        st.session_state["augmented_flag"] = sidebar_container.checkbox("Augmented", value=False, on_change=self.augmented_flag_changed) 
        if st.session_state["augmented_flag"]:
            # show a select box for the dataset selection
            st.session_state["dataset"] = sidebar_container.selectbox("Dataset", constants.DATASETS)
        
        # Set a default model
        if "model" not in st.session_state:
            st.session_state["model"] = constants.MODEL_NAME

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []  

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What's up?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Send message to assistant and display response
            with st.chat_message("assistant"):
                stream = self.pipeline.stream_response(prompt)
            response = st.write_stream(stream)

            # Add assistant response to chat history
            self.pipeline.chat_history.append(prompt)
            self.pipeline.chat_history.append(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.stop()
