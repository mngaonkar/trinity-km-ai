from math import log
from multiprocessing import Pipe
from venv import logger
from sklearn import pipeline
import streamlit as st
from streamlit.components.v1 import html
from langchain_community.chat_models import ChatOllama
from backend.vectorstore import VectorStore
import constants
from langchain_core.language_models import LanguageModelInput
from backend.pipeline import Pipeline
from backend.loader import DocumentLoader
from backend.llm_provider import LLMProvider
import constants
from loguru import logger

class ChatGUI():
    """Chat GUI class for displaying chat messages."""
    def __init__(self, pipeline: Pipeline, vector_store: VectorStore):
        logger.info("Initializing chat GUI...")
        self.pipeline = pipeline
        self.vector_store = vector_store
        self.pipeline.setup_session_state(st.session_state)
        
        logger.info("Chat GUI initialized.")

    def set_vector_store(self, store: VectorStore):
        self.vector_store = store

    def model_changed(self):
        """Model changed."""
        logger.info(f'Setting up large language model {st.session_state["model"]}')
        self.pipeline.llm_provider.set_model(st.session_state["model"])

    def provider_changed(self):
        """Provider changed."""
        pass

    def augmented_flag_changed(self):
        """Augmented flag changed."""
        logger.info(f'Setting up augmentation {st.session_state["augmented_flag"]}')
       
        # self.pipeline.setup(self.vector_store)
    
    def dataset_changed(self):
        """Dataset changed."""
        logger.info(f'Setting up datset {st.session_state["dataset"]}')
        self.pipeline.vector_store.init_vectorstore(st.session_state["dataset"])

    def run(self):
        # Set the page configuration
        st.set_page_config(
            page_title=constants.APP_NAME,
            page_icon=":speech_balloon:",
            layout="wide",
        )

        st.title(constants.APP_NAME + " " + constants.APP_VERSION)
        chat_container = st.container()
        self.sidebar_container = st.sidebar.container()

        # Save use specific settings in session state
        self.pipeline.setup_large_language_model_provider()
        models_list = self.pipeline.llm_provider.get_models_list()
        logger.info(f"Models list: {models_list}")
        self.sidebar_container.selectbox("Model Provider", constants.LLM_PROVIDERS, key="provider", index=0, on_change=self.provider_changed)
        self.sidebar_container.selectbox("Model", models_list, key="model", index=0, on_change=self.model_changed)
        augmented = self.sidebar_container.checkbox("Augmented", value=False, key="augmented_flag", on_change=self.augmented_flag_changed) 
        
        # show a select box for the dataset selection
        if augmented:
            db_files = self.vector_store.database.list_databases()
            self.sidebar_container.selectbox("Dataset", db_files, key="dataset", index=0, on_change=self.dataset_changed)
            self.pipeline.setup(self.vector_store)
        
        # Set a default model
        if "model" not in st.session_state:
            st.session_state["model"] = constants.MODEL_NAME

        # Setup pipeline
        self.pipeline.setup(self.vector_store)

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
