import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the model
with open('./app/models/model2.pkl', 'rb') as file:
    model = pickle.load(file)

with open('./app/models/label_encoder.pkl', 'rb') as file:
    loaded_label_encoder = pickle.load(file)


def dataframe_encoder(df, labelEncoder):
        df_encode = df.copy()

        for col, le in labelEncoder.items():
            print('encoded', df_encode[col])
            df_encode[col] = le.transform(df_encode[col])

        return df_encode

def draw_plots(model, feature_names):
    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top_features = feature_importances.head(10)

    # Plot feature importances
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    top_features.plot(kind='bar')
    plt.title("Top 3 Feature Importance in Predicting ML Experience Level")
    plt.ylabel("Importance Score")
    plt.xlabel("Features")

    st.pyplot(fig2)


def app():

    # Streamlit app title
    st.title("ML Experience Predictor")

    # User inputs
    st.sidebar.header("Input Features")
    education_level = st.sidebar.selectbox(
        "Education Level", ["Master’s degree", "Bachelor’s degree", "Doctoral degree"]
    )
    job_role = st.sidebar.selectbox(
        "Job Role", ["Account Executive", "Administrative Assistant", "Business Analyst"]
    )
    # years_of_experience = st.sidebar.slider("Years of Experience", 0, 20, 5)
    # num_online_courses = st.sidebar.number_input("Number of Online Courses Completed", 0, 50, 5)

    print(education_level, job_role)

    # Prepare the input data
    input_data = pd.DataFrame(
        {
            "CoursesCoursera": ["Coursera"],
            "Education": [education_level],
            "JobTitle": [job_role],
            # "Years_of_Experience": [years_of_experience],
            # "Num_Online_Courses": [num_online_courses],
        }
    )

    feature_names = list(input_data.keys())
    print(input_data)

    encoded_labels = dataframe_encoder(input_data, loaded_label_encoder)

    # Display the input data
    st.write("### Input Features")
    st.write(encoded_labels)

    # Prediction button
    if st.button("Predict ML Experience Level"):
        prediction = model.predict(encoded_labels)
        print('prediction', prediction)

        level = prediction[0]
        level_array = [
            "Entry level",
            "Mid-Senior level",
            "Executive",
            "Internship",
            "Associate",
            "Director",
        ]
        st.write(f"### Predicted ML Experience Level: {level_array[level]}")

        draw_plots(model, feature_names)

    
if __name__ == '__main__':
    app()