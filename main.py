import streamlit as st
import requests
import json
from frontend.gui import GUI
from frontend.chat import ChatGUI

# gui = GUI()
gui = ChatGUI()
gui.run()

