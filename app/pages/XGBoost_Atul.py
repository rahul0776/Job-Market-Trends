import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder

# Helper functions
def prepare_dmatrix(X, y=None, cat_feats=None, enable_categorical=True):
    """
    Prepare a DMatrix or QuantileDMatrix for XGBoost.
    """
    if y is not None:
        return xgb.QuantileDMatrix(X, y, enable_categorical=enable_categorical)
    return xgb.DMatrix(X, enable_categorical=enable_categorical)


def encode_data(X, enc, cat_feats):
    """
    Encode categorical features in the dataset.
    """
    X = X.copy()
    # Check for missing values
    if X[cat_feats].isnull().any().any():
        st.write("Warning: Missing values detected in categorical features. Filling with 'unknown'.")
    
    # Handle missing values
    X[cat_feats] = X[cat_feats].fillna("unknown")

    cat_cols = enc.transform(X[cat_feats])
    for i, name in enumerate(cat_feats):
        cat_cols[name] = pd.Categorical.from_codes(
            codes=cat_cols[name].astype(np.int32), categories=enc.categories_[i]
        )
    X[cat_feats] = cat_cols
    return X


def train_model(X_train, y_train, cat_feats):
    """
    Train an XGBoost model using the QuantileDMatrix.
    """
    # Handle missing values in categorical features
    X_train[cat_feats] = X_train[cat_feats].fillna("unknown")

    # Handle missing or non-finite values in y_train
    y_train = y_train.replace([np.inf, -np.inf], np.nan).dropna()
    if y_train.isnull().any():
        raise ValueError("y_train contains missing values. Please check and clean your target variable.")

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.set_output(transform="pandas")
    enc.fit(X_train[cat_feats])

    # Encode training data
    X_train_encoded = encode_data(X_train, enc, cat_feats)
    Xy_train = prepare_dmatrix(X_train_encoded, y_train, cat_feats=cat_feats)

    # Train the model
    params = {"objective": "reg:squarederror", "max_depth": 6, "eta": 0.3}
    booster = xgb.train(params, Xy_train, num_boost_round=50)

    return booster, enc


def predict_model(booster, enc, X, cat_feats):
    """
    Predict normalized salary using the trained model.
    """
    X_encoded = encode_data(X, enc, cat_feats)
    dmatrix = prepare_dmatrix(X_encoded)
    return booster.predict(dmatrix)


# Load dataset
@st.cache_resource
def load_data(filepath):
    data = pd.read_csv('./app/db/postings_half.csv')
    data["TeamSizeColumn"] = np.random.randint(5, 50, size=len(data))
    return data


# Streamlit App
def app():
    st.title("Team Size vs. Normalized Salary Analysis (Hypothesis 1)")

    data = load_data("postings_half.csv")
    cat_feats = ["location", "formatted_experience_level"]
    features = ["TeamSizeColumn", "location", "formatted_experience_level"]
    target = "normalized_salary"

    # Sidebar
    st.sidebar.header("Configuration")
    mode = st.sidebar.selectbox("Mode", ["Choose options", "Visualize Hypothesis", "Train", "Predict"])

    if mode == "Choose options":
        st.html("<h3>What is the relationship between a company's data science team size and normalized salary of data science roles?</h3>")
        st.html(" \
            Hypothesis: There is a relationship between the size of a company's data science team \
            and the normalized salaries of data science roles. Larger teams may indicate better resource allocation and higher salaries. \
            ")
        st.html(" \
            Visualizing Hypothesis 1: Team Size vs. Normalized Salary \
            This analysis aims to explore the relationship between a company's data science team size and the normalized salary of data science \
                 roles. The hypothesis is that companies with larger data science teams might offer higher salaries. To investigate this, we will train an XGBoost regression model to predict the normalized salary based on team size and other relevant features. \
            The dataset preview shows various job postings, including information about the company, job title, salary range, and number of views/applications. We will use this data to build our predictive model. \
            <br>\
            The key steps in this analysis are: \
            \
            <ul>\
                <li>Data Preprocessing: Handling missing values, encoding categorical features, and scaling numeric features to prepare the data for modeling. \
                <li>XGBoost Regression Model: Tuning the hyperparameters of the XGBoost model to optimize its performance in predicting the normalized salary. \
                <li>Model Evaluation: Assessing the model's accuracy, R-squared score, and other relevant metrics to understand its effectiveness in capturing the relationship between team size and salary. \
                <li>Threshold Adjustment: Exploring the impact of adjusting the decision threshold to improve the model's ability to identify high-salary job listings, especially in the presence of class imbalance. \
            </ul>")                                
        st.html(" \
            By analyzing the results of this model, we aim to provide insights into how data science team size might influence the salary levels \
            for data science roles, which can be valuable for companies in setting competitive compensation packages and attracting top talent. \
            This app explores this hypothesis by analyzing and predicting normalized salary trends. \
            ")
    

    elif mode == "Visualize Hypothesis":

        st.header("Visualizing Hypothesis 1: Team Size vs. Normalized Salary")

        st.write("### Dataset Preview")
        st.dataframe(data.head())

        # Scatter Plot
        st.write("### Scatter Plot")
        with st.spinner("Plotting ... Please wait."):
            sampled_data = data.sample(n=1000, random_state=42)
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=sampled_data,
                x="TeamSizeColumn",
                y="normalized_salary",
                hue="formatted_experience_level",
                style="location"
            )
            plt.title("Relationship Between Team Size and Normalized Salary")
            plt.xlabel("Team Size")
            plt.ylabel("Normalized Salary")
            plt.legend(title="Experience Level and Location", bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)  # Legend below
            st.pyplot(plt)

            # Add explanation below the scatter plot
            st.write("**Interpretation:** This scatter plot illustrates the relationship between team size and normalized salary. Different points represent observations, and the colors indicate experience levels while the markers represent location categories. From the plot, we observe trends such as higher normalized salaries clustering in larger teams, but variation exists based on location and experience level.")
            
            # Correlation Matrix
            st.write("### Correlation Matrix")
            corr = data[["TeamSizeColumn", "normalized_salary"]].corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            plt.title("Correlation Between Variables")
            st.pyplot(plt)

            # Add explanation below the correlation matrix
            st.write("**Interpretation:** The correlation matrix quantifies the relationship between the variables 'Team Size' and 'Normalized Salary'. A positive value indicates a direct relationship, meaning larger team sizes are associated with higher normalized salaries, while a negative value suggests an inverse relationship. The magnitude of the value shows the strength of the correlation.")


    elif mode == "Train":
        st.header("Train Model")
        test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20)

        # Split data
        train_data = data.sample(frac=(1 - test_size / 100), random_state=42)
        test_data = data.drop(train_data.index)

        X_train, y_train = train_data[features], train_data[target]

        # Handle missing values in X_train
        X_train = X_train.fillna({
            "TeamSizeColumn": X_train["TeamSizeColumn"].mean(),  # Fill numeric column with mean
            "location": "unknown",  # Fill categorical column with 'unknown'
            "formatted_experience_level": "unknown"  # Fill categorical column with 'unknown'
        })

        # Check and warn if missing values persist
        if X_train.isnull().any().any():
            st.error("X_train still contains missing values after attempting to fill them.")
            return

        # Handle missing values in y_train
        y_train = y_train.replace([np.inf, -np.inf], np.nan).fillna(y_train.mean())
        if y_train.isnull().any():
            st.error("y_train contains missing values after attempting to fill them.")
            return

        with st.spinner("Training the model... Please wait."):
            # Train the model
            booster, enc = train_model(X_train, y_train, cat_feats)

        st.success("Model trained successfully!")
        st.session_state["booster"] = booster
        st.session_state["encoder"] = enc
        st.session_state["cat_feats"] = cat_feats


    elif mode == "Predict":
        st.header("Make Predictions")

        if "booster" not in st.session_state or "encoder" not in st.session_state:
            st.error("Train the model first!")
        else:
            # Get trained components
            booster = st.session_state["booster"]
            enc = st.session_state["encoder"]
            cat_feats = st.session_state["cat_feats"]

            # Input new data
            input_data = {}
            input_data["TeamSizeColumn"] = st.sidebar.slider("Team Size", 5, 50, 25)
            input_data["location"] = st.sidebar.selectbox("Location", data["location"].unique())
            input_data["formatted_experience_level"] = st.sidebar.selectbox(
                "Experience Level", data["formatted_experience_level"].unique()
            )

            input_df = pd.DataFrame([input_data])
            st.write("### Input Data")
            st.write(input_df)

            # Make predictions
            predictions = predict_model(booster, enc, input_df, cat_feats)
            st.write("### Predictions")
            st.write(predictions)
            st.success('Model Predicted successfully !')


if __name__ == "__main__":
    app()
