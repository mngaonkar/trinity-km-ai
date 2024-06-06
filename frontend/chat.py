from multiprocessing import Pipe
from venv import logger
import streamlit as st
from streamlit.components.v1 import html
from langchain_community.chat_models import ChatOllama
import constants
from langchain_core.language_models import LanguageModelInput
from backend.pipeline import Pipeline
from backend.loader import DocumentLoader
from backend.llm import LLM
import constants
from loguru import logger

class ChatGUI():
    def __init__(self):
        # self.chat = LLM(local_llm=constants.MODEL_NAME, base_url=constants.INFERENCE_URL)
        self.chat = Pipeline()
        self.loader = DocumentLoader()
        self.init_vectorstore()

    def init_vectorstore(self):
        logger.info("Initializing vector store")
        # webpage = "https://www.nifty.org/nifty/bisexual/adult-friends/debauchery-of-a-young-indian-housewife/debauchery-of-a-young-indian-housewife-"
       
        # for i in range(1, 6):
        #     url = webpage + str(i)
        #     docs = self.loader.load_web_document(url)
        #     self.chat.db.vectorstore.add_documents(docs)
        
        # webpage = "https://gutenberg.org/cache/epub/1661/pg1661.txt"
        # docs = self.loader.load_web_document(webpage)
        # self.chat.db.vectorstore.add_documents(docs)

        doc_location = constants.DOCS_LOCATION
        docs = self.loader.load_documents_from_directory(doc_location)
        logger.info(f"Adding {len(docs)} documents to vector store")
        self.chat.db.vectorstore.add_documents(docs)
        logger.info("Done.")

    def run(self):
        # Set the page configuration
        st.set_page_config(
            page_title="Trinity AI",
            page_icon=":speech_balloon:",
            layout="wide",
        )

        st.title(constants.APP_NAME + " " + constants.APP_VERSION)
        chat_container = st.container()
        
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

        if prompt := st.chat_input("What is up?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                stream = self.chat.stream_response(prompt)
            response = st.write_stream(stream)

            # Add assistant response to chat history
            self.chat.chat_history.append(prompt)
            self.chat.chat_history.append(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
