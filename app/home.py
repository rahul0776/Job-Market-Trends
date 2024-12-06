import streamlit as st
from pages import Database_Operations
from pages import XGBoost_Atul, Support_Vector_Regressor_Rahul, Neural_Network_Atharva, Random_Forest_Classifier_Abhinav

# Set the page configuration
st.set_page_config(page_title="Future Job Market Trends", layout="wide")

# Home Page Content
st.title("Modeling Future Job Market Trends")
st.subheader("Using University Enrollment, Industry Data, and Other Factors")

# Motivation Section
st.markdown(
    """
    ### Motivation
    The job market is highly volatile, influenced by factors such as:
    - Technological advancements
    - Industry supply and demand
    - External events like war and natural disasters
    
    **Objective:** Help individuals make informed decisions about pursuing relevant educational foundations and identifying suitable job markets.
    """
)

# Dataset Section
st.markdown(
    """
    ### Datasets
    - **University employment data:** Scraped from university websites.
    - **Government and public sources:** APIs from organizations like the US Bureau of Labor Statistics.
    """
)

# Group Members Section
st.markdown(
    """
    ### Group Members
    - *Abhinav Tembulkar* (50602510) - [Email](mailto:atembulk@buffalo.edu)
    - *Atul Pandey* (50594507) - [Email](mailto:atulpriy@buffalo.edu)
    - *Atharva Prabhu* (50591634) - [Email](mailto:aprabhu5@buffalo.edu)
    - *Rahul Lotlikar* (50604152) - [Email](mailto:rahulujv@buffalo.edu)
    """
)

# Phase 3 Questions Section
st.markdown(
    """
    ### Phase 3 Questions
    - **Abhinav Tembulkar**:
      - How effective are online learning platforms in improving job market readiness compared to traditional university degrees?
    - **Atharva Prabhu**:
      - Is it possible to predict expected salary range or employment status based on education major and demographic factors?
    - **Atul Pandey**:
      - What is the relationship between a company's data science team size and normalized salary of data science roles?
    - **Rahul Lotlikar**:
      - Can we predict an individual's job level within an organization based on age, experience, and education?
    """
)

# Footer
st.markdown(
    """
    ---
    **Code and detailed explanation:** Check the provided Jupyter Notebook or visit the 
    [GitHub Repository](https://github.com/APrabhu21/DIC_Project).
    """
)