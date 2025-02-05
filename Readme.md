# Modeling Future Job Market Trends Using University Enrolment, Industry Data and other factors.

### Website
https://jobnomix.streamlit.app/

The branch 'merge_fix' has the latest updated code used for deployment.

## Motivation 
The job market is very volatile, easily affected by multiple factors such as Technological advancements, industry supply and demand, and external events like War and natural catastrophes. With this project, we aim to help people make an informed decision on choosing suitable job market and pursue relevant educational foundation. 

## Datasets 
Data  from multiple university employment data, government and public websites. Data will be obtained via scraping on University employment data websites. Data via API from government and public websites such as US Bureau of Labor Statistics, etc will also be obtained.




### Questions:

Q)How effective are online learning platforms in improving job market readiness compared to traditional university degrees ? 
<br>

Q) Given an individual's educational major and other demographic factors, it is possible to predict their expected salary range or employment status?
<br>

Q) What is the relationship between the size of a company's data science team and the normalized salary of data science-related roles? 
 <br>



Q)Can we accurately predict an individual's job level within an organization based on demographic and professional characteristics such as age, experience level, and education?
<br>

## Build Steps
Building app steps : 
Steps to Build and Deploy a Streamlit App with Conda

1. Install Conda
If you haven’t already, install Conda from Miniconda or Anaconda. All commands are to be run in root folder ./DIC_Project

2. Set Up the Conda Environment
    -  	Create a new Conda environment:
            `conda create -n streamlit_env python=3.12`
    -   Activate the environment:
            `conda activate streamlit_env`
    -   Install Streamlit and dependencies:
            `pip install -r requirements.txt`

3. Run the App Locally
    -	Start the app:
            `streamlit run app/home.py`
    -	Open the URL provided (e.g., http://localhost:8501) in your browser.
    -   Alternatively, you can find the deployed website on :
            `https://jobnomix.streamlit.app/`

## Folder structure

This repository is organized as follows:

```plaintext
📂 app/
   ├── 📂 db/                 # Contains database-related files
   ├── 📂 models/             # Stores machine learning models
   ├── 📂 pages/              # Pages with each team members deployed model 
   └── home.py                # Main Streamlit app entry point

📂 data/               # Preprocessed datasets generated after phase2 EDA
📂 datasets/           # Raw Datasets used in the phase1
📂 docs/               # Documentation files related to data



📂 preprocessing/             # Preprocessing scripts and utilities



.gitignore                # Git ignore file for excluding unnecessary files
requirements.txt          # Python dependencies for the project
README.md                 # Project documentation
```
