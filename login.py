import streamlit as st
import constants
from streamlit_google_auth import Authenticate

class Login():
    def __init__(self) -> None:
        self.authenticator = Authenticate(
                secret_credentials_path='/Users/mahadev/Downloads/client_secret_255024374055-7p2p3pjh1usib8pu0k0a6vn7josvj4bm.apps.googleusercontent.com.json',
                cookie_name='my_cookie_name',
                cookie_key='this_is_secret',
                redirect_uri='https://console.altbox.one:8501',
            )

    def run(self):
        st.title("Login with Google")
        if self.authenticator.check_authentification():
            st.write("You are logged in")
        else:
            if st.button("Login"):
                 self.authenticator.login()
        
        if not st.session_state['connected']:
            st.image(st.session_state['user_info'].get('picture'))
            st.write(f"Hello, {st.session_state['user_info'].get('name')}")
            st.write(f"Your email is {st.session_state['user_info'].get('email')}")
            if st.button('Log out'):
                self.authenticator.logout()


if __name__ == "__main__":
    login = Login()
    login.run()