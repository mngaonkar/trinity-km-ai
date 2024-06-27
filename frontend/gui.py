import streamlit as st
from streamlit.components.v1 import html
from langchain_community.chat_models import ChatOllama
from backend.llm import LLM
import constants

class GUI():
    def __init__(self):
        self.chat = LLM(local_llm=constants.MODEL_NAME, base_url=constants.INFERENCE_URL_OLLAMA)

    def run(self):
        st.set_page_config(
        page_title="Trinity AI",
        page_icon=":robot_face:",
        layout="wide",
        )
        
        st.title(constants.APP_NAME + " " + constants.APP_VERSION)

        # create sidebar
        st.sidebar.write("Navigation")
  
        # Create buttons in the sidebar
        if st.sidebar.button("Home"):
            selected_page = "Home"
        elif st.sidebar.button("About"):
            selected_page = "About"
        elif st.sidebar.button("Contact"):
            selected_page = "Contact"
        elif st.sidebar.button("Settings"):
            selected_page = "Settings"
        else:
            selected_page = "Home"  # Default page

        # prompt form
        with st.form(key='my_form'):
            text = st.text_area('Ask question', 'what is the advice for learning AI?', height=200)
            submit_button = st.form_submit_button(label='Submit')

        # response text
        response_text = st.empty()

        # update response text
        def update_response_text(content):
            html_content = f'<div class="fixed-width">{content}</div>'
            response_text.markdown(html_content, unsafe_allow_html=True)
        
        css = """
        <style>
        .fixed-width {
            width: 1000px;
            margin: 0 auto;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            text-align: left;
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

        # submit button action handler
        if submit_button:
                self.chat.generate_response(text, update_response_text)


        