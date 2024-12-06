import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt


# Helper Functions
def prepare_dmatrix(X, y=None, cat_feats=None, enable_categorical=True):
    if y is not None:
        return xgb.QuantileDMatrix(X, y, enable_categorical=enable_categorical)
    return xgb.DMatrix(X, enable_categorical=enable_categorical)


def encode_data(X, enc, cat_feats):
    X = X.copy()
    cat_cols = enc.transform(X[cat_feats])
    for i, name in enumerate(cat_feats):
        cat_cols[name] = pd.Categorical.from_codes(
            codes=cat_cols[name].astype(np.int32), categories=enc.categories_[i]
        )
    X[cat_feats] = cat_cols
    return X


def train_model(X_train, y_train, cat_feats):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    enc.set_output(transform="pandas")
    enc.fit(X_train[cat_feats])

    # Encode training data
    X_train_encoded = encode_data(X_train, enc, cat_feats)
    Xy_train = prepare_dmatrix(X_train_encoded, y_train, cat_feats=cat_feats)

    # Train XGBoost model
    params = {"objective": "reg:squarederror", "max_depth": 6, "eta": 0.3}
    booster = xgb.train(params, Xy_train, num_boost_round=50)

    return booster, enc


def predict_model(booster, enc, X, cat_feats):
    X_encoded = encode_data(X, enc, cat_feats)
    dmatrix = prepare_dmatrix(X_encoded)
    return booster.predict(dmatrix)


# Streamlit App
def app():
    st.title("XGBoost with Hypothesis 1 Visualization")

    # Load a demo dataset
    n_samples = 200
    data = pd.DataFrame(
        {
            "team_size": np.random.randint(1, 50, size=n_samples),
            "normalized_salary": np.random.uniform(50_000, 120_000, size=n_samples),
            "location": np.random.choice(["NY", "CA", "TX", "WA"], n_samples),
            "experience_level": np.random.choice(["junior", "mid-level", "senior"], n_samples),
        }
    )
    data["label"] = data["normalized_salary"] * np.random.uniform(0.8, 1.2, size=n_samples)

    cat_feats = ["location", "experience_level"]
    features = cat_feats + ["team_size"]
    target = "label"

    # Sidebar
    st.sidebar.header("Configuration")
    mode = st.sidebar.selectbox("Mode", ["Train", "Predict", "Visualize Hypothesis 1"])

    if mode == "Train":
        st.header("Train Model")
        test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20)

        # Split data
        train_data = data.sample(frac=(1 - test_size / 100), random_state=42)
        test_data = data.drop(train_data.index)

        X_train, y_train = train_data[features], train_data[target]

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
            booster = st.session_state["booster"]
            enc = st.session_state["encoder"]
            cat_feats = st.session_state["cat_feats"]

            # Input new data
            input_data = {}
            for feat in cat_feats:
                input_data[feat] = st.sidebar.selectbox(f"{feat}", data[feat].unique())
            input_data["team_size"] = st.sidebar.slider("Team Size", 1, 50, 10)

            input_df = pd.DataFrame([input_data])
            st.write("### Input Data")
            st.write(input_df)

            # Make predictions
            predictions = predict_model(booster, enc, input_df, cat_feats)
            st.write("### Predictions")
            st.write(predictions)

    elif mode == "Visualize Hypothesis 1":
        st.header("Hypothesis 1: Relationship Between Team Size and Normalized Salary")
        st.write(
            """
            This visualization explores the relationship between team size and normalized salary
            to identify potential trends. Companies with larger teams may offer competitive salaries.
            """
        )

        # Scatter plot with regression line
        st.write("### Scatter Plot with Regression Line")
        plt.figure(figsize=(10, 6))
        sns.regplot(data=data, x="team_size", y="normalized_salary")
        plt.title("Team Size vs. Normalized Salary")
        plt.xlabel("Team Size")
        plt.ylabel("Normalized Salary")
        st.pyplot(plt)

        # Box plot by experience level
        st.write("### Box Plot by Experience Level")
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data, x="experience_level", y="normalized_salary")
        plt.title("Normalized Salary by Experience Level")
        plt.xlabel("Experience Level")
        plt.ylabel("Normalized Salary")
        st.pyplot(plt)


if __name__ == "__main__":
    app()
