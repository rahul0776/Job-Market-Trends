import streamlit as st
import pandas as pd
import os
import time

# Path to your CSV file (update this path)
# CSV_FILE = "./app/db/imputed_decoded_dataset.csv"
DB_PATH = "./app/db/"

# Load the CSV file into a DataFrame
# @st.cache_data
def load_data(CSV_FILE):
    CHOSEN_FILE_PATH = os.path.join(DB_PATH, CSV_FILE)
    return pd.read_csv(CHOSEN_FILE_PATH), CHOSEN_FILE_PATH

# Save the DataFrame back to the CSV file
def save_data(df, FILE_PATH):
    df.to_csv(FILE_PATH, index=False)

def list_files():
    files = os.listdir(DB_PATH)
    print(files)
    files = [file for file in files if len(file.split('.')) == 2 and file.split('.')[1] == 'csv']
    return files

# App functionality for the database page
def app():
    st.title("Manage Database (CSV)")
    
    # Select db
    st.sidebar.title("Select Databases")
    file_list = list_files()
    print(file_list)
    file_chosen = st.sidebar.selectbox("Select DB", file_list)
    print(file_chosen)
    
    df = None
    CURRENT_FILE = None
    
    # Select action
    st.sidebar.title("Actions")
    action = st.sidebar.selectbox("Select Action", ["View", "Add Entry", "Modify Entry", "Remove Entry"])

    with st.spinner("Loading data... Please wait."):
        # Load the data
        df, CURRENT_FILE = load_data(file_chosen)
        
        # Show the current database
        st.write("### Current Database: ", f"{file_chosen}")
        st.dataframe(df)

    if action == "View":
        st.write("### View Records")
        st.write("Use filters below to search specific records.")
        filters = {}
        
        # Add filters for each column
        for col in df.columns:
            unique_values = ["All"] + df[col].dropna().unique().tolist()
            filters[col] = st.selectbox(f"Filter by {col}", unique_values)
        
        # Filter the DataFrame
        filtered_df = df
        for col, value in filters.items():
            if value != "All":
                filtered_df = filtered_df[filtered_df[col] == value]
        
        # Display filtered data
        st.write("### Filtered Results")
        st.dataframe(filtered_df)

    elif action == "Add Entry":
        st.write("### Add New Entry")
        new_data = {}
        
        # Create input fields for each column
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                new_data[col] = st.number_input(f"Enter value for {col}")
            else:
                new_data[col] = st.text_input(f"Enter value for {col}")
        
        if st.button("Add Entry"):
            # Append the new row and save
            # df = df.append(new_data, ignore_index=True)
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            save_data(df, CURRENT_FILE)
            st.success("New entry added successfully!")
            time.sleep(1)
            st.rerun()

    elif action == "Modify Entry":
        st.write("### Modify Existing Entry")
        st.write("Select a record to modify.")
        
        # Select the row to modify
        row_index = st.number_input("Enter row index to modify", min_value=0, max_value=len(df) - 1, step=1)
        selected_row = df.iloc[row_index]
        st.write("Selected Row:")
        st.write(selected_row)

        st.write("Press submit button after making changes")

        # Create input fields for modification
        updated_data = {}

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                updated_data[col] = st.number_input(f"Modify {col}", value=float(selected_row[col]))
            else:
                updated_data[col] = st.text_input(f"Modify {col}", value=str(selected_row[col]))
        
        if st.button("Save Changes"):
            # Update the DataFrame
            for col in df.columns:
                df.at[row_index, col] = updated_data[col]
            
            save_data(df, CURRENT_FILE)
            st.success("Entry modified successfully!")
            time.sleep(1)
            st.rerun()

    elif action == "Remove Entry":
        st.write("### Remove Existing Entry")
        st.write("Select a record to remove.")
        
        # Select the row to delete
        row_index = st.number_input("Enter row index to delete", min_value=0, max_value=len(df) - 1, step=1)
        if st.button("Remove Entry"):
            # Remove the row and save
            df = df.drop(index=row_index).reset_index(drop=True)
            save_data(df, CURRENT_FILE)
            st.success("Entry removed successfully!")
            time.sleep(1)
            st.rerun()

if __name__ == '__main__':
    app()