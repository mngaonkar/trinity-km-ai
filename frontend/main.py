import streamlit as st
import requests
import json

INFERENCE_URL = "http://10.0.0.147:8080/completion"

st.title("Trinity v1.0")
st.write("I am a bot, how can I help you today?")
user_input = st.text_input("You: ", "")

if st.button("Submit"):
    st.write(f"sending {user_input}")
    session = requests.Session()
    response_text = st.empty()

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
    def update_response_text(content):
        html_content = f'<div class="fixed-width">{content}</div>'
        response_text.markdown(html_content, unsafe_allow_html=True)


    response = session.post(INFERENCE_URL, json={"prompt": user_input, "n_predict": 512, "stream":True}, stream=True)
    full_text = ""
    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                full_text += json.loads(chunk.decode("utf-8").split("data:")[1])["content"]
                update_response_text(full_text)
    else:
        st.write("Bot: Sorry, I am not available right now.")

