# Modeling Future Job Market Trends Using University Enrolment, Industry Data and other factors.

### Website
https://jobnomix.streamlit.app/

The branch 'merge_fix' has the latest updated code used for deployment.

## Motivation
The job market is very volatile, easily affected by multiple factors such as Technological advancements, industry supply and demand, and external events like War and natural catastrophes. With this project, we aim to help people make an informed decision on choosing suitable job market and pursue relevant educational foundation. 

## Datasets 
Data from multiple university employment data, government and public websites. Data will be obtained via scraping on University employment data websites. Data via API from government and public websites such as US Bureau of Labor Statistics, etc will also be obtained.

## Group members
Abhinav Tembulkar - 50602510 - atembulk@buffalo.edu
Atul Pandey - 50594507 - atulpriy@buffalo.edu
Atharva Prabhu - 50591634 - aprabhu5@buffalo.edu
Rahul Lotlikar - 50604152 - rahulujv@buffalo.edu

## Additional requirements for phase 2
### Questions
Abhinav Tembulkar - 50602510 -<br>
Q)How effective are online learning platforms in improving job market readiness compared to traditional university degrees ? File:50602510_algo.ipynb, From cell 20
<br>

Atharva Prabhu - 50591634 -<br>
Q) Given an individual's educational major and other demographic factors, it is possible to predict their expected salary range or employment status? File:50591634_phase2.ipynb, From cell 7
<br>
Atul Pandey - 50594507 -<br>
Q) What is the relationship between the size of a company's data science team and the normalized salary of data science-related roles? File:50594507_phase2.ipynb, From Cell 2
 <br>


Rahul Lotlikar - 50604152 -<br>
Q)Can we accurately predict an individual's job level within an organization based on demographic and professional characteristics such as age, experience level, and education? File:50604152AlgosPhase2.ipynb. From Cell 19
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

📂 exp/
   ├── 50602510_algo.ipynb    # Final version of phase 2 notebook for 50602510
   └── SVRVisualization.ipynb # Final version of phase 2 notebook for 

📂 preprocessing/             # Preprocessing scripts and utilities

📂 src/
   ├── 50591634_eda_new.ipynb     # Phase 1 EDA notebook
   ├── 50591634_phase2.ipynb      # Phase 2 notebook 
   ├── 50594507_phase2.ipynb      # Phase 2 notebook 
   ├── 50594507_Proj1.ipynb       # Phase 1 EDA notebook
   ├── 50602510_algo.ipynb        # Phase 2 notebook
   ├── 50602510_eda.ipynb         # Phase 1 EDA notebook
   ├── 50604152_eda.ipynb         # Phase 1 EDA notebook
   └── 50604152AlgosPhase2.ipynb  # Phase 2 notebook 
   └── DIC_common_data_notebook.ipynb  # Common notebook for shared data analysis Phase 1

.gitignore                # Git ignore file for excluding unnecessary files
requirements.txt          # Python dependencies for the project
README.md                 # Project documentation
```
