import streamlit as st
from langchain_community.chat_models import ChatOllama

st.title("Trinity v1.0")
chat = ChatOllama(model="llama2", base_url="http://10.0.0.147:11434", temperature=0)

def generate_response(prompt):
    return chat.invoke(prompt)

with st.form(key='my_form'):
    text = st.text_area('Enter text', 'what is the advice for learning AI?', height=200)
    submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        response = generate_response(text)
        st.info(response.content)