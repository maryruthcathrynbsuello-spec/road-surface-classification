import streamlit as st

def render_sidebar():
    pages = [

        st.Page(
            "views/dashboard.py", 
            title="Dashboard", 
            icon=":material/dashboard:",
            default=True
        ),

        st.Page(
            "views/dataset_overview.py", 
            title="Dataset Overview",
            icon=":material/dataset:"
        ),

        st.Page(
            "views/upload.py", 
            title="Upload",
            icon=":material/upload:"
        ),

        st.Page(
            "views/model_performance.py", 
            title="Model Performance",
            icon=":material/analytics:"
        ),

        st.Page(
            "views/history.py", 
            title="History",
            icon=":material/history:"
        ),

        st.Page(
            "views/settings.py", 
            title="Settings",
            icon=":material/settings:"
        ),

    ]

    return st.navigation(pages, position="sidebar")