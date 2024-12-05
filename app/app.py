import streamlit as st
from pages import abhi_page, atharva_page, rahul_page, atul_page

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Page 1", "Page 2", "Page 3", "Page 4"])
st.text('Choose any page to go see that algorithm')

# Render the selected page
if page == "Page 1":
    abhi_page.app()
elif page == "Page 2":
    atharva_page.app()
elif page == "Page 3":
    rahul_page.app()
elif page == "Page 4":
    atul_page.app()