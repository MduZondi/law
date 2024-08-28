import streamlit as st

# Set up Streamlit page configuration
st.set_page_config(page_title='AiVocator Junior Counsel', page_icon='AIvocator_icon.png', layout="wide")

# Apply dark theme
st.markdown("""
    <style>
    .stApp { background-color: #1E1E1E; color: #FFFFFF; }
    .stButton>button { color: #FFFFFF; background-color: #4CAF50; }
    .stTextInput>div>div>input { color: #FFFFFF; background-color: #2E2E2E; }
    .stTextArea>div>div>textarea { color: #FFFFFF; background-color: #2E2E2E; }
    </style>
""", unsafe_allow_html=True)

# Homepage content
st.title('AI-powered Junior Counsel ⚖️')
st.write("Revolutionize your legal practice with cutting-edge AI. Make informed decisions, streamline your workflow, and stay ahead of the competition.")

st.markdown("---")
st.write("Select a tool from the sidebar to get started.")
