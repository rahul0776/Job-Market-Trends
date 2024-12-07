import streamlit as st
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the trained SVR model and scaler
with open('./app/models/rahul_svr_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('./app/models/rahul_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the column order used during training
TRAINED_COLUMNS = [
    'Age',
    'ExperienceLevel_Director',
    'ExperienceLevel_Entry level',
    'ExperienceLevel_Executive',
    'ExperienceLevel_Internship',
    'ExperienceLevel_Mid-Senior level',
    'Education_Doctoral degree',
    'Education_I prefer not to answer',
    'Education_Master’s degree',
    'Education_No formal education past high school',
    'Education_Professional degree',
    'Education_Some college/university study without earning a bachelor’s degree'
]


def plot_feature_contributions(features, contributions):
    """
    Creates a bar plot to visualize feature contributions.

    Args:
    - features: List of feature names.
    - contributions: List of feature contribution values.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=features, y=contributions, palette="viridis")
    plt.title("Feature Contributions to Predicted Job Level")
    plt.ylabel("Contribution Value")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(plt)


def job_level(title):
    title = title.lower()  # Convert to lowercase for consistency
    if "intern" in title or "junior" in title or "assistant" in title:
        return 1  # Entry-Level
    elif "analyst" in title or "associate" in title or "specialist" in title:
        return 2  # Mid-Level
    elif "senior" in title or "lead" in title or "manager" in title:
        return 3  # Senior-Level
    elif "director" in title or "head" in title or "vp" in title or "vice president" in title:
        return 4  # Executive-Level
    elif "c-level" in title or "chief" in title or "officer" in title:
        return 5  # C-Level
    else:
        return 2  # Default to Mid-Level if unknown


def convert_age(value):
    if '-' in value:
        start, end = map(int, value.split('-'))
        return (start + end) / 2
    elif value == '70+':
        return 75
    else:
        return pd.to_numeric(value, errors='coerce')


def train(df):
    # Apply job level function to categorize job titles
    df['JobLevel'] = df['JobTitle'].apply(job_level)

    # Select relevant columns including additional features
    data = df[['Age', 'ExperienceLevel', 'Education', 'JobLevel']].copy()

    data['Age'] = data['Age'].apply(convert_age)

    # Drop any rows with missing values in 'Age' or 'JobLevel'
    data = data.dropna(subset=['Age', 'JobLevel', 'ExperienceLevel', 'Education'])

    # Step 3: One-hot encode categorical variables like 'ExperienceLevel' and 'Education'
    data = pd.get_dummies(data, columns=['ExperienceLevel', 'Education'], drop_first=True)

    # Define features and target variable
    X = data.drop(columns='JobLevel')
    y = data['JobLevel']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize SVR with RBF kernel
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    # Train the model
    svr.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = svr.predict(X_test_scaled)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return svr, (mae, mse, rmse, r2)


def app():
    # Add the heading question
    st.markdown("### Can we accurately predict an individual's job level within an organization based on demographic and professional characteristics such as age, experience level, and education?")

    # Sidebar
    st.sidebar.header("Configuration")
    mode = st.sidebar.selectbox("Mode", ["Choose Options", "Train", "Predict"])

    if mode == "Train":
        df = pd.read_csv('./app/db/imputed_decoded_dataset2.csv')
        
        with st.spinner("Training the model... Please wait."):
            trained_model, metrics = train(df)
            with open('./app/models/rahul_svr_model.pkl', 'wb') as file3:
                pickle.dump(trained_model, file3)

        st.success('Model successfully trained')
        mae, mse, rmse, r2 = metrics

        st.write("Mean Absolute Error (MAE)")
        st.write(mae)
        st.write("Mean Squared Error (MSE)")
        st.write(mse)
        st.write("Root Mean Squared Error (RMSE)")
        st.write(rmse)
        st.write("R² Score")
        st.write(r2)


    elif mode == "Predict":
        # Input features
        st.sidebar.header("Input Features")
        st.sidebar.markdown("Enter the features below:")

        # User Inputs
        age = st.sidebar.slider("Age", 18, 75, 30)
        education_level = st.sidebar.selectbox(
            "Education Level", [
                "Doctoral degree",
                "I prefer not to answer",
                "Master’s degree",
                "No formal education past high school",
                "Professional degree",
                "Some college/university study without earning a bachelor’s degree"
            ]
        )
        experience_level = st.sidebar.selectbox(
            "Experience Level", [
                "Director",
                "Entry level",
                "Executive",
                "Internship",
                "Mid-Senior level"
            ]
        )

        # One-hot encode the categorical inputs
        input_data = pd.DataFrame(columns=TRAINED_COLUMNS)
        input_data.loc[0] = 0  # Initialize all columns to 0

        # Set the values for the user input
        input_data['Age'] = age
        input_data[f'Education_{education_level}'] = 1
        input_data[f'ExperienceLevel_{experience_level}'] = 1

        # Display input data
        st.write("### Input Data")
        st.write(input_data)

        # Scale the input data
        try:
            # Pass the raw numpy array to avoid feature name issues
            scaled_data = scaler.transform(input_data.values)
            st.write("### Scaled Data")
            st.write(scaled_data)

            # Predict button
            if st.button("Predict Job Level"):
                prediction = model.predict(scaled_data)
                st.write(f"### Predicted Job Level: {round(prediction[0], 2)}")

                # Calculate and display feature contributions
                feature_contributions = scaled_data[0] * model.coef_ if hasattr(model, 'coef_') else scaled_data[0]
                plot_feature_contributions(TRAINED_COLUMNS, feature_contributions)
        except Exception as e:
            st.error(f"Error scaling input data: {e}")


if __name__ == '__main__':
    app()


        
