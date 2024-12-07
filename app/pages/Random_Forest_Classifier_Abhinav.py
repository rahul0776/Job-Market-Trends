import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def load_files():
    # Load the model
    with open('./app/models/abhi_rfc_model.pkl', 'rb') as file1:
        model = pickle.load(file1)

    with open('./app/models/abhi_label_encoder.pkl', 'rb') as file2:
        loaded_label_encoder = pickle.load(file2)

    return model, loaded_label_encoder


def dataframe_encoder(df, labelEncoder):
        df_encode = df.copy()

        for col, le in labelEncoder.items():
            # print('encoded', df_encode[col])
            df_encode[col] = le.transform(df_encode[col])

        return df_encode


def draw_plots(model, feature_names):
    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top_features = feature_importances.head(10)

    # Plot feature importances
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    top_features.plot(kind='bar')
    plt.title("Top Feature Importances in Predicting ML Experience Level")
    plt.ylabel("Importance Score")
    plt.xlabel("Features")

    st.pyplot(fig2)
    st.write('##### The graph demonstrates that Online courses are not very significant in predicting experience level in Machine Learning field')
    st.write('##### Other factors such as Median salary are more important in predicting experience. Thus from our model, online platforms are not very effective in job market readiness')


def detect_and_ordinal_encode(df):
    # Create a copy of the original dataframe
    df_encoded = df.copy()
    
    # Initialize a dictionary to store label encoders for each column
    label_encoders = {}

    # Loop through columns
    for col in df_encoded.columns:
        # Check if column is categorical (object type)
        if df_encoded[col].dtype == 'object':
            # Apply Label Encoding
            le = LabelEncoder()
            non_na_mask = df_encoded[col].notna()  # Mask for non-NaN values
            df_encoded.loc[non_na_mask, col] = le.fit_transform(df_encoded.loc[non_na_mask, col].astype(str))
            # df_encoded[col] = df_encoded[col].astype(str).apply(lambda x: le.fit_transform([x]) if pd.notna(x) else x)
            
            # Store the LabelEncoder for future use (e.g., for decoding or applying to test data)
            label_encoders[col] = le

    return df_encoded, label_encoders


def train(df):
    # Selecting relevant features
    # Assuming 'MLExperienceYears' is the target variable and 'CoursesPlatform', 'Education', 'JobTitle' are relevant 
    # Modify these column names based on your actual data

    features = ['CoursesPlatform', 'Education', 'JobTitle', 'SalaryMedian']  # Add or modify columns based on available relevant features
    X = df[features]
    y = df['MLExperienceYears']  # Target variable

    i = 0
    experience_mapping = {}
    y.loc[y == 0] = '< 1 years'
    for k in y.unique():
        experience_mapping[k] = i
        i+=1
    y = y.map(experience_mapping)

    encoded_X , labels_encoders = detect_and_ordinal_encode(X)
    X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.3, random_state=42)


    # Initialize the Random Forest model
    rfc_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    # Train the model
    rfc_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = rfc_model.predict(X_test)
    y_test_arr = y_test.to_numpy()

    # Evaluate the model
    accuracy = accuracy_score(y_test_arr, y_pred)
    classification_rep = classification_report(y_test_arr, y_pred)
    conf_matrix = confusion_matrix(y_test_arr, y_pred)

    with open('./app/models/abhi_label_encoder.pkl', 'wb') as file:
        pickle.dump(labels_encoders, file)

    metrics = (accuracy, classification_rep, conf_matrix)
    return rfc_model, metrics


def app():

    # Streamlit app title
    st.markdown("### How effective are online learning platforms in improving job market readiness compared \
                to traditional university degrees ?")

    # Sidebar
    st.sidebar.header("Configuration")
    mode = st.sidebar.selectbox("Mode", ["Choose Options", "Train", "Predict", "Visualize"])

    if mode == "Choose Options":
        st.html("\
            <b>The Random Forest Classifier (RFC) was employed to analyze how online learning \
             platforms enhance job market readiness by improving machine learning experience among \
             professionals. This model, with its ensemble of decision trees, effectively handles \
            imbalanced datasets and provides robust predictions without overfitting. By leveraging \
            the class_weight='balanced' parameter, RFC addressed skewed distributions in experience \
            levels, ensuring fair representation across categories</b>")
        st.html("<b> \
                Additionally, feature \
                importance analysis revealed that online courses and educational qualifications are key \
                determinants of professional growth, underscoring the value of up-to-date, practical \
                learning. The model's effectiveness was validated using a confusion matrix and feature \
                importance metrics, supporting the hypothesis that online platforms significantly \
                contribute to career advancement in machine learning roles.</b>")


    elif mode == "Train":
        df = pd.read_csv('./app/db/imputed_decoded_dataset.csv')
        
        with st.spinner("Training the model... Please wait."):
            trained_model, metrics = train(df)
            with open('./app/models/abhi_rfc_model.pkl', 'wb') as file3:
                pickle.dump(trained_model, file3)

        st.success('Model successfully trained')
        accuracy, classification_rep, conf_matrix = metrics

        st.write("Testing accuracy")
        st.write(accuracy * 100)
        st.write("Classification Report")
        st.write(classification_rep)
        
        fig, ax = plt.subplots(figsize=(3, 3))  # Create a proper figure and axis
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp.plot(cmap=plt.cm.Blues, ax=ax)  # Pass the axis explicitly to ConfusionMatrixDisplay
        ax.set_title("Confusion Matrix")  # Set your title on the axis
        st.pyplot(fig)


    elif mode == "Predict":

        loaded_model, loaded_label_encoder = load_files()

        # User inputs
        st.sidebar.header("Input Features")
        education_level = st.sidebar.selectbox(
            "Education Level", ["Master’s degree", "Bachelor’s degree", "Doctoral degree"]
        )
        job_role = st.sidebar.selectbox(
            "Job Role", ["Account Executive", "Administrative Assistant", "Business Analyst"]
        )
        course_platform = st.sidebar.selectbox(
            "Course Platform", ["Coursera", "EdX", "Udemy"]
        )
        salary = st.sidebar.slider("Salary", 60000, 130000, step=100)
        # print(education_level, job_role)

        # Prepare the input data
        input_data = pd.DataFrame(
            {
                "CoursesPlatform": [course_platform],
                "Education": [education_level],
                "JobTitle": [job_role],
                "SalaryMedian": [salary]
            }
        )

        feature_names = list(input_data.keys())
        # print(input_data)

        encoded_labels = dataframe_encoder(input_data, loaded_label_encoder)

        # Display the input data
        st.write("### Input Features")
        st.write(encoded_labels)

        # Prediction button
        if st.button("Predict ML Experience Level"):
            prediction = loaded_model.predict(encoded_labels)
            # print('prediction', prediction)

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
            draw_plots(loaded_model, feature_names)
            st.success('Model predicted succesfully')


    elif mode == "Visualize":
        df = pd.read_csv('./app/db/imputed_decoded_dataset.csv')
        st.write("### Value Counts of MLExperienceYears")

        # Plotting the bar chart
        fig, ax = plt.subplots(figsize=(3, 3))  # Create a figure and axis
        df['MLExperienceYears'].value_counts().plot(
            kind='bar', edgecolor='black', ax=ax
        )  # Plot the value counts on the axis
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.set_title('Value Counts of MLExperienceYears')

        # Display the plot in Streamlit
        st.pyplot(fig)
        st.write("As we can see 'MLExperienceYears' catagorical column is highly imbalanced. \
                 Hence, we used Random Forest Classifier with 'class_weight=balanced' to tackle \
                 highly imbalanced distribution of experience ranges.")
    
if __name__ == '__main__':
    app()