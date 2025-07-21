import base64
import streamlit as st
from streamlit_option_menu import option_menu
import about
import account
import contact_form

st.set_page_config(page_title="TollTracker.com", page_icon="🚗", layout="wide")

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# img = get_img_as_base64("images\itemize.jpg")

page_bg_img = f"""
<style>
/* Main content background with transparency */
[data-testid="stAppViewContainer"] {{
    # background: url("https://i.pinimg.com/originals/db/bd/cd/dbbdcd04cb3f236167333267e4713130.jpg");
    background-color: black;
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color:white;
}}

[data-testid="stAppViewContainer"] > .main {{
    background: rgba(0, 0, 0, 0.4); /* White background with some transparency to let the image slightly show through */
    padding: 20px;
    border-radius: 10px;
    color:white;
}}

/* Sidebar background with transparency */
[data-testid="stSidebar"] > div:first-child {{
    background-color: #6bb367;
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
   
}}

[data-testid="stSidebar"] > div {{
    background: rgba(255, 255, 255, 0.7); /* Sidebar background with some transparency */
    padding: 10px;
    border-radius: 10px;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, function):
        self.apps.append({
            "title": title,
            "function": function
        })

    def run(self):
        with st.sidebar:
            app = option_menu(
                menu_title='TollTracker',
                options=['Account', 'About Us', 'Contact Us'],
                icons=['person-circle', 'info-circle-fill', 'envelope'],
                menu_icon='cast',
                default_index=0,
                styles={
                    "container": {"padding": "2!important", "background-color": '#242524'},
                    "icon": {"color": "white", "font-size": "20px"},
                    "nav-link": {"color": "white", "font-size": "18px", "text-align": "left", "margin": "5px", "--hover-color": "#8bc68e"},
                    "nav-link-selected": {"background-color": "#6bb367"},
                } 
            )


        if app == "Account":
            account.app()
        if app == 'About Us':
            about.app()
        if app == "Contact Us":
            contact_form.app()

app = MultiApp()
app.run()
