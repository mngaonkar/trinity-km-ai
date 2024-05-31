import streamlit as st
from streamlit.components.v1 import html
from langchain_community.chat_models import ChatOllama
import constants
from langchain_core.language_models import LanguageModelInput
from backend.llm import LLM

class ChatGUI():
    def __init__(self):
        self.chat = LLM(local_llm=constants.MODEL_NAME, base_url=constants.INFERENCE_URL)

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
            st.session_state.messages.append({"role": "assistant", "content": response})
